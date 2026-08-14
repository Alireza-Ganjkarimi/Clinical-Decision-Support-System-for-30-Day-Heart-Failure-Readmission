import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import warnings
import streamlit.components.v1 as components
# --- LangChain and LLM Modules ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

warnings.filterwarnings('ignore')

# ==========================================
# 0. Proxy & API Settings (For LLM module)
# ==========================================
PROXY_URL = "http://127.0.0.1:10809" 
os.environ['http_proxy'] = PROXY_URL
os.environ['https_proxy'] = PROXY_URL
os.environ['HTTP_PROXY'] = PROXY_URL
os.environ['HTTPS_PROXY'] = PROXY_URL

os.environ["GROQ_API_KEY"] = "Enter your API key here"

# ==========================================
# 1. General Page Settings (Frontend Config)
# ==========================================
st.set_page_config(page_title="HF Readmission CDSS", layout="wide", page_icon="🫀")

# ==========================================
# 2. Load Architecture Components (Backend & LLM)
# ==========================================
@st.cache_resource
def load_artifacts():
    try:
        package = joblib.load('hf_deployment_package.pkl')
        return package
    except FileNotFoundError:
        st.error("File 'hf_deployment_package.pkl' not found. Please run the model saving code first.")
        st.stop()

@st.cache_data
def load_lab_references():
    try:
        labs = pd.read_csv('labevents.csv', usecols=['itemid', 'valueuom', 'ref_range_lower', 'ref_range_upper'])
        labs = labs.dropna(subset=['ref_range_lower', 'ref_range_upper'])
        
        # Filter out incorrect ranges
        labs = labs[labs['ref_range_upper'] > labs['ref_range_lower']]
        labs = labs.drop_duplicates(subset=['itemid'])
        
        d_labs = pd.read_csv('d_labitems.csv', usecols=['itemid', 'label'])
        merged = labs.merge(d_labs, on='itemid', how='left')
        
        ref_dict = {}
        for _, row in merged.iterrows():
            label = row['label']
            if pd.notna(label):
                ref_dict[label] = {
                    'lower': row['ref_range_lower'],
                    'upper': row['ref_range_upper'],
                    'unit': row['valueuom'] if pd.notna(row['valueuom']) else ""
                }
        return ref_dict
    except Exception:
        return {}

@st.cache_data
def load_mock_ehr():
    try:
        # File containing all extracted guideline-driven features
        return pd.read_csv('hf_guideline_driven_features.csv')
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_patient_labs(sub_id, max_time_str):
    d_labs = pd.read_csv('d_labitems.csv', usecols=['itemid', 'label'])
    
    chunks = []
    for chunk in pd.read_csv('labevents.csv', usecols=['subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum'], chunksize=1000000):
        patient_chunk = chunk[chunk['subject_id'] == sub_id]
        if not patient_chunk.empty:
            chunks.append(patient_chunk)
            
    if not chunks:
        return pd.DataFrame()
        
    patient_labs = pd.concat(chunks)
    patient_labs['charttime'] = pd.to_datetime(patient_labs['charttime'])
    max_time = pd.to_datetime(max_time_str)
    
    patient_labs = patient_labs[patient_labs['charttime'] <= max_time]
    patient_labs = patient_labs.merge(d_labs, on='itemid', how='left')
    patient_labs = patient_labs.dropna(subset=['valuenum'])
    patient_labs = patient_labs.sort_values(by='charttime')
    
    return patient_labs

# --- LLM Module Initialization (Only once on page load) ---
@st.cache_resource
def init_llm_rag_chain():
    PDF_FILE_NAME = "heidenreich-et-al-2022-2022-aha-acc-hfsa-guideline-for-the-management-of-heart-failure-a-report-of-the-american-college.pdf" 
    CHROMA_DB_DIR = "./chroma_db_api_version"
    
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name="pritamdeka/S-PubMedBert-MS-MARCO")
    
    if not os.path.exists(CHROMA_DB_DIR):
        loader = PyPDFLoader(PDF_FILE_NAME)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300, separators=["\n\n", "\n", ".", " ", ""])
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=CHROMA_DB_DIR)
    else:
        vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)

    #retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    retriever = vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={
            "k": 10,            
            "fetch_k": 50,
            "lambda_mult": 0.75  
        } 
    )
    
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.1, 
        max_tokens=2048
    )
    

    system_prompt = (
    "You are an expert clinical cardiologist assistant and Clinical Decision Support System (CDSS) analyzer. "
    "Your analysis MUST be strictly grounded in the provided 2022 AHA/ACC/HFSA Heart Failure guidelines context. "
    "Do NOT invent, hallucinate, or recommend clinical interventions outside of this context. "
    "The user will provide patient clinical parameters (LVEF, HR, BP, labs, comorbidities, current medications) "
    "and a Logistic Regression 30-day readmission risk prediction percentage and status. "
    "Generate a highly professional, evidence-based clinical report divided EXACTLY into these 3 specific sections:\n\n"
    
    "### 1. Clinical and Risk Summary\n"
    "Briefly synthesize the patient's admission-to-discharge trajectory, vital signs, and key biomarkers (e.g., NT-proBNP, Potassium, Creatinine). "
    "State their calculated 30-day readmission risk status and interpret what this means for their immediate care.\n\n"
    
    "### 2. GDMT Optimization (Based on 2022 Guidelines)\n"
    "Systematically assess the 4 pillars of Guideline-Directed Medical Therapy (GDMT) based on the patient's LVEF classification (HFrEF, HFmrEF, HFpEF, or HFimpEF) and vital/lab parameters:\n"
    "- **Renin-Angiotensin System Inhibition:** Check for ACEi/ARB or ARNi. Note that ARNi is preferred over ACEi/ARB to further reduce morbidity and mortality in symptomatic HFrEF, provided BP permits.\n"
    "- **Beta Blockers:** Assess if one of the evidence-based beta blockers is prescribed and optimized based on the patient's discharge heart rate and blood pressure.\n"
    "- **Mineralocorticoid Receptor Antagonists (MRAs):** Evaluate if an MRA is indicated. Ensure serum potassium is < 5.0 mEq/L and renal function is adequate before recommending.\n"
    "- **SGLT2 Inhibitors:** Recommend initiation of SGLT2i for symptomatic chronic HFrEF to reduce hospitalization and mortality, irrespective of the presence of type 2 diabetes.\n"
    "- **Diuretics:** Assess the need for loop diuretics to relieve congestion based on fluid retention and urine output history.\n\n"
    
    "### 3. Discharge & Follow-up Action Plan\n"
    "Provide concrete, actionable steps for transition of care. Include:\n"
    "- Recommendations for adjusting diuretics based on volume status.\n"
    "- Requirements for monitoring safety labs (e.g., renal function and electrolytes) 1 to 2 weeks after GDMT initiation/titration.\n"
    "- A mandate for an early follow-up outpatient visit (e.g., within 7 days of discharge) given the readmission risk.\n\n"
    
    "Context from Guideline:\n{context}"
)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

package = load_artifacts()
model = package['model']
scaler = package['scaler']
threshold = package['threshold']
model_features = package['features'] 

ref_dict = load_lab_references()
mock_ehr = load_mock_ehr()
rag_chain = init_llm_rag_chain() 

# ==========================================
# 3. Calculate Global Means for All Features
# ==========================================
global_means = {}
if not mock_ehr.empty:
    global_means = mock_ehr.mean(numeric_only=True).to_dict()

# For model features, use Scaler means exactly to avoid disrupting math calculations
if hasattr(scaler, 'mean_'):
    for feat, m in zip(model_features, scaler.mean_):
        global_means[feat] = m

meta_cols = ['subject_id', 'hadm_id', 'admittime', 'dischtime', 'readmission_30d']
all_features = [c for c in mock_ehr.columns if c not in meta_cols]
other_features = [f for f in all_features if f not in model_features]

# --- Sorting logic for sidebar fields ---
def feature_sort_key(feat):
    if feat.startswith('drug_'):
        return (0, 'Medication', feat)
    elif feat.startswith('comorb_'):
        return (1, 'Comorbidity', feat)
    elif feat.startswith('bp_'):
        return (2, 'Blood Pressure', feat)
    elif feat.startswith('icu_HR'):
        return (3, 'Heart Rate', feat)
    elif feat.startswith('icu_LVEF'):
        return (4, 'LVEF', feat)
    elif feat.startswith('icu_urine'):
        return (5, 'Urine', feat)
    elif feat.startswith('lab_'):
        parts = feat.split('_', 2)
        if len(parts) == 3:
            return (6, parts[2], parts[1]) # Group by lab name, then sort by first/last/delta
        return (6, feat, '')
    else:
        return (7, feat, '')

other_features = sorted(other_features, key=feature_sort_key)

# ==========================================
# 4. Header and Central UI Design
# ==========================================
st.title("Interactive Clinical Decision Support System")
st.markdown("""
**30-Day Heart Failure Readmission Risk Prediction & Guideline-Based Clinical Assessment**  
---
""")

# ==========================================
# 5. Sidebar Design & EHR Mockup Search
# ==========================================
st.sidebar.header("🔍 EHR Record Search (Mockup)")
search_sub_id = st.sidebar.text_input("Patient Subject ID (subject_id):")
search_hadm_id = st.sidebar.text_input("Admission ID (hadm_id):")

prefill_data = None
if search_sub_id and search_hadm_id:
    try:
        sub_id_val = int(search_sub_id)
        hadm_id_val = int(search_hadm_id)
        
        if not mock_ehr.empty:
            match = mock_ehr[(mock_ehr['subject_id'] == sub_id_val) & (mock_ehr['hadm_id'] == hadm_id_val)]
            if not match.empty:
                prefill_data = match.iloc[0]
                st.sidebar.success("✅ Patient Data Auto-Filled Successfully!")
            else:
                st.sidebar.error("⚠️ Record not found. Please fill manually.")
    except ValueError:
        st.sidebar.error("⚠️ Please enter numeric IDs only.")

st.sidebar.markdown("---")

# Helper function to translate variable names into fluent medical terms
def get_clinical_label(feat):
    clinical_labels = {
        'drug_BetaBlocker': 'Beta Blocker',
        'drug_ACE_ARB_ARNI': 'ACEi / ARB / ARNi',
        'drug_MRA': 'Mineralocorticoid Receptor Antagonist (MRA)',
        'drug_LoopDiuretic': 'Loop Diuretic',
        
        'comorb_AFib': 'Atrial Fibrillation',
        'comorb_Diabetes': 'Diabetes Mellitus',
        'comorb_Hypertension': 'Hypertension',
        'comorb_CKD': 'Chronic Kidney Disease',
        
        'icu_HR_last': 'Heart Rate at Discharge (bpm)',
        'icu_HR_mean': 'Mean Heart Rate at ICU (bpm)',
        'icu_HR_min': 'Minimum Heart Rate at ICU (bpm)',
        'icu_HR_max': 'Maximum Heart Rate at ICU (bpm)',
        'icu_LVEF_last': 'Left Ventricular Ejection Fraction (LVEF %)',
        'icu_urine_output_total': 'Total Urine Output (mL)',
        
        'bp_SBP_first': 'Systolic Blood Pressure at Admission (mmHg)',
        'bp_SBP_last': 'Systolic Blood Pressure at Discharge (mmHg)',
        'bp_SBP_min': 'Minimum Systolic Blood Pressure (mmHg)',
        'bp_SBP_max': 'Maximum Systolic Blood Pressure (mmHg)',
        'bp_DBP_last': 'Diastolic Blood Pressure at Discharge (mmHg)',
        'bp_SBP_delta': 'Systolic Blood Pressure Change (mmHg)',
    }
    
    if feat in clinical_labels:
        return clinical_labels[feat]
        
    if feat.startswith('lab_'):
        parts = feat.split('_', 2)
        if len(parts) == 3:
            stat = parts[1]
            test = parts[2]
            
            unit_suffix = " (mg/dL)" if test in ['BUN', 'Urea Nitrogen'] else ""
            
            if stat == 'first':
                return f"{test} at Admission{unit_suffix}"
            elif stat == 'last':
                return f"{test} at Discharge{unit_suffix}"
            elif stat == 'delta':
                return f"{test} Change (Discharge - Admission){unit_suffix}"
                
    return feat

def render_input_field(feat):
    clinical_name = get_clinical_label(feat)
    
    # For medications and comorbidities (Changing history to 0 and 1 buttons for medical staff convenience)
    if 'drug_' in feat.lower() or 'comorb_' in feat.lower():
        if prefill_data is not None and feat in prefill_data:
            default_val = int(prefill_data[feat]) if pd.notna(prefill_data[feat]) else 0
        else:
            default_val = int(global_means.get(feat, 0)) 
            
        idx = 1 if default_val == 1 else 0
        
        prompt_text = f"Prescribe {clinical_name}:" if 'drug_' in feat.lower() else f"History of {clinical_name}:"
        return st.sidebar.radio(prompt_text, options=[0, 1], index=idx, horizontal=True)
        
    # For lab tests and vital signs (Numeric fields)
    else:
        label_str = clinical_name
        if 'lab_' in feat and 'delta' not in feat:
            parts = feat.split('_', 2)
            if len(parts) == 3:
                test_name = parts[2]
                ref_info = ref_dict.get(test_name, None)
                if ref_info and pd.notna(ref_info['lower']) and pd.notna(ref_info['upper']):
                    unit_str = f" {ref_info['unit']}" if ref_info['unit'] else ""
                    label_str = f"{clinical_name} (Normal: {ref_info['lower']} - {ref_info['upper']}{unit_str})"
                    
        # Default value: Priority to patient data, then global average
        if prefill_data is not None and feat in prefill_data:
            default_val = float(prefill_data[feat]) if pd.notna(prefill_data[feat]) else float(global_means.get(feat, 0.0))
        else:
            default_val = float(global_means.get(feat, 0.0))
            
        return st.sidebar.number_input(label_str, value=default_val, step=0.1)

user_inputs = {}

# Sidebar Section 1: Logistic Regression Model Features
st.sidebar.header("⚙️ Logistic Regression Model Features")
st.sidebar.markdown("These directly impact the risk percentage:")
for feat in model_features:
    user_inputs[feat] = render_input_field(feat)

# Sidebar Section 2: Supplemental Clinical Features (for LLM and Guidelines)
st.sidebar.markdown("---")
with st.sidebar.expander("📋 Other Guideline-Driven Clinical Features"):
    st.markdown("Contextual data for Guidelines & LLM Analysis:")
    for feat in other_features:
        user_inputs[feat] = render_input_field(feat)

# Create input dataframe for the model (only 5 main features to avoid errors)
input_df_all = pd.DataFrame([user_inputs])
model_input_df = input_df_all[model_features] 
input_scaled = scaler.transform(model_input_df)

# ==========================================
# 6. Prediction Engine and Alarm System
# ==========================================
risk_prob = model.predict_proba(input_scaled)[0, 1]
risk_percentage = risk_prob * 100
threshold_percentage = threshold * 100

# Define a safety margin for early warning (e.g., 5% below the threshold)
safety_margin = 5.0
medium_risk_lower_bound = max(0.0, threshold_percentage - safety_margin)

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 📈 Readmission Risk within 30 days:")
    st.metric(label="Probability", value=f"{risk_percentage:.1f} %")
    st.markdown(f"*Clinical Threshold (Cut-off): **{threshold_percentage:.1f}%***")

with col2:
    st.markdown("### 🚨 Recommended Clinical Decision:")
    
    if risk_percentage >= threshold_percentage:
        # Red Alert: Crossed the official clinical threshold
        st.error(
            f"**High Risk Patient (Action Required)**\n\n"
            f"🚨 The patient's risk ({risk_percentage:.1f}%) has **crossed the clinical threshold ({threshold_percentage:.1f}%)**. "
            "Requires referral to a multidisciplinary heart failure disease management program, "
            "GDMT optimization, and an early follow-up visit within 7 days."
        )
    elif risk_percentage >= medium_risk_lower_bound:
        # Yellow Alert: Close to threshold (Mitigates False Negatives / Early Warning)
        st.warning(
            f"**Moderate / Borderline Risk (Early Warning)**\n\n"
            f"⚠️ The patient's risk ({risk_percentage:.1f}%) is close to the threshold (Safety Margin: {medium_risk_lower_bound:.1f}% - {threshold_percentage:.1f}%). "
            "Carefully address precipitating causes and high-risk comorbidities, and optimize Guideline-Directed Medical Therapy (GDMT) before discharge."
        )
    else:
        # Green Alert: Safe
        st.success(
            f"**Low Risk Patient**\n\n"
            f"✅ The patient's condition is relatively stable ({risk_percentage:.1f}%). "
            "Provide patient-centered discharge instructions, ensure a clear transitional care plan, and continue optimized GDMT."
        )

st.markdown("---")

# ==========================================
# 7. Explainability Engine (XAI - Local SHAP)
# ==========================================
st.subheader("🔍 Root Cause Analysis (Logistic Regression Factors)")
st.markdown("Shows how the key model features influenced the risk percentage.")

background_scaled = np.zeros((1, len(model_features)))
explainer = shap.Explainer(model.predict_proba, background_scaled, feature_names=[get_clinical_label(f) for f in model_features])
shap_values = explainer(input_scaled)

explanation = shap_values[..., 1][0]
# Replace scaled data with raw values for easier interpretation
explanation.data = model_input_df.iloc[0].values 

fig, ax = plt.subplots(figsize=(10, 6))
shap.plots.waterfall(explanation, show=False, max_display=10)
st.pyplot(fig)

st.markdown("---")

# ==========================================
# 8. Physiological Stability Radar Chart
# ==========================================
st.subheader("🕸️ Physiological Stability (Guideline Biomarkers)")
st.markdown("This chart evaluates key guideline-directed laboratory values against the **Normal Reference Range**.", unsafe_allow_html=True)

radar_labels = []
radar_patient_vals = []
# --- Lists to store the actual values for tooltip visualization ---
actual_patient_vals = []
actual_lower_vals = []
actual_upper_vals = []

# Vital HF guideline biomarkers
vital_guideline_labs = ['Sodium', 'Potassium', 'Creatinine', 'BUN', 'Hemoglobin', 'NTproBNP']

for test_name in vital_guideline_labs:
    ref_info = ref_dict.get(test_name, None)
    if ref_info and pd.notna(ref_info['lower']) and pd.notna(ref_info['upper']):
        val = None
        # Strict priority to the latest test to plot only one sample
        if f"lab_last_{test_name}" in user_inputs:
            val = user_inputs[f"lab_last_{test_name}"]
        elif f"lab_first_{test_name}" in user_inputs:
            val = user_inputs[f"lab_first_{test_name}"]
            
        if val is not None:
            lower = ref_info['lower']
            upper = ref_info['upper']
            if upper > lower:
                midpoint = (upper + lower) / 2.0
                half_range = (upper - lower) / 2.0
                norm_val = 2.0 + ((val - midpoint) / half_range)
                norm_val = max(0.0, min(4.5, norm_val)) 
                
                radar_labels.append(f"{test_name}")
                radar_patient_vals.append(norm_val)
                # Store original values
                actual_patient_vals.append(val)
                actual_lower_vals.append(lower)
                actual_upper_vals.append(upper)

# Add hemodynamic indicators (HR and BP) with manual clinical normal ranges
manual_vitals = {
    'Systolic BP': {'val': user_inputs.get('bp_SBP_last', None), 'lower': 90, 'upper': 130},
    'Heart Rate': {'val': user_inputs.get('icu_HR_last', None), 'lower': 60, 'upper': 100}
}

for name, info in manual_vitals.items():
    if info['val'] is not None and pd.notna(info['val']):
        lower, upper = info['lower'], info['upper']
        midpoint = (upper + lower) / 2.0
        half_range = (upper - lower) / 2.0
        norm_val = 2.0 + ((info['val'] - midpoint) / half_range)
        norm_val = max(0.0, min(4.5, norm_val))
        
        radar_labels.append(name)
        radar_patient_vals.append(norm_val)
        # Store original values
        actual_patient_vals.append(info['val'])
        actual_lower_vals.append(lower)
        actual_upper_vals.append(upper)

if radar_labels:
    # Close the polygon for radar chart
    radar_labels.append(radar_labels[0])
    radar_patient_vals.append(radar_patient_vals[0])
    actual_patient_vals.append(actual_patient_vals[0])
    actual_lower_vals.append(actual_lower_vals[0])
    actual_upper_vals.append(actual_upper_vals[0])
    
    fig_radar = go.Figure()
    
    # Lower Normal Limit
    fig_radar.add_trace(go.Scatterpolar(
        r=[1]*len(radar_labels), 
        theta=radar_labels, 
        mode='lines', 
        line=dict(color='rgba(46, 204, 113, 0.8)', dash='dash'), 
        name='Lower Normal Limit',
        customdata=actual_lower_vals,
        hovertemplate='%{theta}: %{customdata}<extra>Lower Limit</extra>'
    ))
    
    # Upper Normal Limit
    fig_radar.add_trace(go.Scatterpolar(
        r=[3]*len(radar_labels), 
        theta=radar_labels, 
        mode='lines', 
        fill='tonext', 
        fillcolor='rgba(46, 204, 113, 0.2)', 
        line=dict(color='rgba(46, 204, 113, 0.8)', dash='dash'), 
        name='Upper Normal Limit',
        customdata=actual_upper_vals,
        hovertemplate='%{theta}: %{customdata}<extra>Upper Limit</extra>',
        hoveron='points'
    ))
    
    # Patient Profile
    fig_radar.add_trace(go.Scatterpolar(
        r=radar_patient_vals, 
        theta=radar_labels, 
        mode='lines+markers', 
        line=dict(color='rgba(231, 76, 60, 1)', width=3), 
        marker=dict(size=8, color='rgba(231, 76, 60, 1)'), 
        name='Patient Profile',
        customdata=actual_patient_vals,
        hovertemplate='%{theta}: %{customdata}<extra>Patient Value</extra>'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, 
                range=[0, 4.5], 
                tickvals=[1, 2, 3], 
                ticktext=['Min Normal', 'Ideal', 'Max Normal'], 
                gridcolor='rgba(0,0,0,0.1)'
            )
        ), 
        showlegend=True, 
        margin=dict(l=80, r=80, t=40, b=40)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ==========================================
# 9. Historical Laboratory Trends
# ==========================================
if prefill_data is not None:
    st.markdown("---")
    st.subheader("📉 Historical Guideline Biomarker Trends")
    st.markdown("Displays the trajectory of key guideline-directed biomarkers up to the discharge time.")
    with st.spinner("Loading historical lab data..."):
        # Add important trend features based on guidelines (e.g., infection and glucose)
        target_lab_names = {'Sodium', 'Potassium', 'Creatinine', 'BUN', 'Hemoglobin', 'NTproBNP', 'Glucose', 'White Blood Cells'}
        disch_time_str = prefill_data['dischtime']
        sub_id_val = int(prefill_data['subject_id'])
        patient_labs = load_patient_labs(sub_id_val, disch_time_str)
        
        if not patient_labs.empty:
            patient_labs_filtered = patient_labs[patient_labs['label'].isin(target_lab_names)]
            if not patient_labs_filtered.empty:
                unique_labs = sorted(patient_labs_filtered['label'].unique())
                cols = st.columns(2)
                for i, lab_name in enumerate(unique_labs):
                    lab_data = patient_labs_filtered[patient_labs_filtered['label'] == lab_name]
                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(x=lab_data['charttime'], y=lab_data['valuenum'], mode='lines+markers', name='Raw History', line=dict(color='#3498db', width=2), marker=dict(size=6, color='#2980b9')))
                    unit_info = ref_dict.get(lab_name, {}).get('unit', '')
                    unit_str = f" ({unit_info})" if unit_info else ""
                    fig_trend.update_layout(title=dict(text=f"{lab_name} Trend", font=dict(size=14)), xaxis=dict(title="Date/Time", tickformat="%Y-%m-%d", showgrid=True), yaxis=dict(title=f"Result Value{unit_str}", showgrid=True), margin=dict(l=40, r=40, t=40, b=40), height=320, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    with cols[i % 2]:
                        st.plotly_chart(fig_trend, use_container_width=True)

# ==========================================
# 10. Generative AI Module (Guideline-Based RAG)
# ==========================================
st.markdown("---")
st.subheader("🤖 LLM Clinical Reasoning & Guideline Analysis")
st.markdown("Generate a clinical report analyzing the patient's data against the 2022 AHA/ACC/HFSA guidelines.")

if st.button("Generate Guideline-Based Clinical Report", type="primary"):
    with st.spinner("Analyzing patient profile against guidelines..."):
        
        # Convert patient features to a structured text string for the prompt
        patient_profile_str = "\n".join([f"{get_clinical_label(k)}: {v}" for k, v in user_inputs.items()])
        
        # Determine exact 3-tier risk status for the LLM
        if risk_percentage >= threshold_percentage:
            risk_status = "High Risk (Action Required)"
        elif risk_percentage >= medium_risk_lower_bound:
            risk_status = "Moderate / Borderline Risk (Early Warning)"
        else:
            risk_status = "Low Risk (Stable)"
        
        # Inject exact mathematical threshold context into the prompt
        llm_query = f"""
        --- Model Risk Assessment ---
        Patient's Calculated Risk: {risk_percentage:.1f}%
        Risk Status: {risk_status}
        
        *Context for LLM*: The model's strict High-Risk cutoff is {threshold_percentage:.1f}%. The Moderate-Risk (early warning) zone is between {medium_risk_lower_bound:.1f}% and {threshold_percentage:.1f}%. Do NOT assume a low absolute percentage (e.g., 20%) means low risk if it is near or above these clinical thresholds. Always align your recommendations with the assigned 'Risk Status'.
        
        --- Patient Clinical Data ---
        {patient_profile_str}
        
        Please evaluate this patient's profile and provide the 3-section report as instructed.
        """
        
        try:
            response = rag_chain.invoke({"input": llm_query})  
            # Context Safety Check 
            retrieved_docs = response.get("context", [])
            if len(retrieved_docs) == 0:
                st.warning("⚠️ No relevant guidelines found for this patient's condition (or the similarity threshold is too high). To prevent hallucination/misinformation, the AI report was aborted.")
            else:
                st.markdown("#### 📄 AI-Generated Clinical Report:")
                st.write(response["answer"])
            # Display retrieved sources (optional for medical transparency)
            with st.expander("📚 References Extracted from Guidelines"):
                for i, doc in enumerate(retrieved_docs):
                    st.caption(f"**Source {i+1}:** {doc.page_content[:200]}...")
        except Exception as e:
            st.error(f"Error communicating with Groq API: {e}")
    # ==========================================
    # 11. Export & Print Options
    # ==========================================
st.markdown("---")

export_buttons_html = """
<div style="display: flex; gap: 15px; margin-top: 20px; justify-content: flex-start;">
    <button id="btn-pdf" style="background-color: #e74c3c; color: white; padding: 8px 16px; border: none; border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: bold; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        📄 Save as PDF
    </button>
    <button id="btn-print" style="background-color: #2980b9; color: white; padding: 8px 16px; border: none; border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: bold; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        🖨️ Print Dashboard
    </button>
</div>

<script>
    document.getElementById('btn-pdf').addEventListener('click', function() {
        window.print();
    });
    
    document.getElementById('btn-print').addEventListener('click', function() {
        window.print();
    });
</script>
"""
st.html(export_buttons_html, unsafe_allow_javascript=True)