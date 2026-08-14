import pandas as pd
import numpy as np

# This code extracts the features mentioned in the 2022 AHA/ACC/HFSA guidelines from the demo version of the MIMIC-IV dataset for 30-day heart failure readmission risk prediction.

# 1. Load base file
print("Loading base file...")
base_cohort = pd.read_csv('hf_base_cohort.csv')
base_cohort['admittime'] = pd.to_datetime(base_cohort['admittime'])
base_cohort['dischtime'] = pd.to_datetime(base_cohort['dischtime'])

# ==========================================
# 2. Extract strictly guideline-based lab tests
# ==========================================
print("Extracting guideline biomarkers...")
labs = pd.read_csv('labevents.csv')

# Key lab test IDs based on guidelines
guideline_lab_ids = [50963, 50912, 51006, 50983, 50971, 51222] 
lab_names = {
    50963: 'NTproBNP',
    50912: 'Creatinine',
    51006: 'BUN',
    50983: 'Sodium',
    50971: 'Potassium',
    51222: 'Hemoglobin'
}

labs = labs[labs['itemid'].isin(guideline_lab_ids)]
labs = labs.dropna(subset=['valuenum'])
labs['label'] = labs['itemid'].map(lab_names)

labs_grouped = labs.groupby(['hadm_id', 'label'])['valuenum'].agg(['first', 'last']).reset_index()
labs_grouped['delta'] = labs_grouped['last'] - labs_grouped['first']

labs_pivot = labs_grouped.pivot(index='hadm_id', columns='label', values=['first', 'last', 'delta'])
labs_pivot.columns = [f"lab_{stat}_{test}" for stat, test in labs_pivot.columns]
labs_pivot = labs_pivot.reset_index()


# ==========================================
# 3. Extract Blood Pressure from OMR
# ==========================================
print("Extracting blood pressure from OMR table...")
omr = pd.read_csv('omr.csv')
bp_data = omr[omr['result_name'].str.contains('Blood Pressure', case=False, na=False)].copy()
bp_data[['SBP', 'DBP']] = bp_data['result_value'].str.split('/', expand=True)
bp_data['SBP'] = pd.to_numeric(bp_data['SBP'], errors='coerce')
bp_data['DBP'] = pd.to_numeric(bp_data['DBP'], errors='coerce')
bp_data = bp_data.dropna(subset=['SBP', 'DBP'])
bp_data['chartdate'] = pd.to_datetime(bp_data['chartdate'])

bp_merged = pd.merge(bp_data, base_cohort[['subject_id', 'hadm_id', 'admittime', 'dischtime']], on='subject_id', how='inner')
bp_valid = bp_merged[
    (bp_merged['chartdate'].dt.date >= bp_merged['admittime'].dt.date) & 
    (bp_merged['chartdate'].dt.date <= bp_merged['dischtime'].dt.date)
].copy()
bp_valid = bp_valid.sort_values(by=['hadm_id', 'chartdate', 'seq_num'])

bp_grouped = bp_valid.groupby('hadm_id').agg(
    bp_SBP_first=('SBP', 'first'),   
    bp_SBP_last=('SBP', 'last'),     
    bp_SBP_min=('SBP', 'min'),       
    bp_SBP_max=('SBP', 'max'),       
    bp_DBP_last=('DBP', 'last')      
).reset_index()
bp_grouped['bp_SBP_delta'] = bp_grouped['bp_SBP_last'] - bp_grouped['bp_SBP_first']


# ==========================================
# 4. Extract guideline-based comorbidities
# ==========================================
print("Extracting Comorbidities...")
diagnoses = pd.read_csv('diagnoses_icd.csv')
diagnoses['icd_code'] = diagnoses['icd_code'].astype(str)

conditions = {
    'comorb_AFib': diagnoses['icd_code'].str.startswith(('42731', 'I48')),
    'comorb_Diabetes': diagnoses['icd_code'].str.startswith(('250', '249', 'E08', 'E09', 'E10', 'E11', 'E13')),
    'comorb_Hypertension': diagnoses['icd_code'].str.startswith(('401', '405', 'I10', 'I15')),
    'comorb_CKD': diagnoses['icd_code'].str.startswith(('585', 'N18'))
}

for col_name, condition in conditions.items():
    patients_with_cond = diagnoses[condition]['subject_id'].unique()
    base_cohort[col_name] = base_cohort['subject_id'].isin(patients_with_cond).astype(int)


# ==========================================
# 5. Extract GDMT medications
# ==========================================
print("Processing GDMT medications...")
drugs = pd.read_csv('prescriptions.csv')
drugs['drug'] = drugs['drug'].astype(str).str.lower()

gdmt_keywords = {
    'drug_BetaBlocker': ['metoprolol', 'carvedilol', 'bisoprolol'],
    'drug_ACE_ARB_ARNI': ['lisinopril', 'enalapril', 'captopril', 'losartan', 'valsartan', 'sacubitril'],
    'drug_MRA': ['spironolactone', 'eplerenone'],
    'drug_LoopDiuretic': ['furosemide', 'torsemide', 'bumetanide']
}

for col_name, keywords in gdmt_keywords.items():
    pattern = '|'.join(keywords)
    hadm_with_drug = drugs[drugs['drug'].str.contains(pattern, na=False)]['hadm_id'].unique()
    base_cohort[col_name] = base_cohort['hadm_id'].isin(hadm_with_drug).astype(int)


# ==========================================
# 6. Initial merging of hospital data
# ==========================================
print("Merging hospital data...")
final_df = base_cohort.merge(labs_pivot, on='hadm_id', how='left')
final_df = final_df.merge(bp_grouped, on='hadm_id', how='left')


# ==========================================
# 7. Extract urine output from Outputevents (ICU ward)
# ==========================================
print("Extracting urine output (for congestion evaluation)...")
try:
    oe = pd.read_csv('outputevents.csv')
    oe['value'] = pd.to_numeric(oe['value'], errors='coerce')
    
    # Calculate total urine or fluid output volume during ICU stay
    urine_grouped = oe.groupby('hadm_id').agg(
        icu_urine_output_total=('value', 'sum')
    ).reset_index()
    
    final_df = final_df.merge(urine_grouped, on='hadm_id', how='left')
except Exception as e:
    print("Error processing outputevents:", e)


# ==========================================
# 8. Extract Heart Rate and LVEF from Chartevents (ICU ward)
# ==========================================
print("Extracting Heart Rate and LVEF...")
try:
    # 220045 = Routine Heart Rate
    # 227008 = Ejection Fraction (LVEF) 
    ce = pd.read_csv('chartevents.csv')
    ce_filtered = ce[ce['itemid'].isin([220045, 227008])].copy()
    ce_filtered['valuenum'] = pd.to_numeric(ce_filtered['valuenum'], errors='coerce')
    ce_filtered = ce_filtered.dropna(subset=['valuenum'])
    ce_filtered = ce_filtered.sort_values(by=['hadm_id', 'charttime'])
    
    # A) Heart Rate
    hr_data = ce_filtered[ce_filtered['itemid'] == 220045]
    hr_grouped = hr_data.groupby('hadm_id').agg(
        icu_HR_last=('valuenum', 'last'),    # Heart rate before discharge/end of ICU stay
        icu_HR_mean=('valuenum', 'mean'),    # Mean heart rate
        icu_HR_min=('valuenum', 'min'),      # Minimum heart rate (Bradycardia detection)
        icu_HR_max=('valuenum', 'max')       # Maximum heart rate (Tachycardia detection/need for beta-blocker up-titration)
    ).reset_index()
    
    # B) Ejection Fraction (LVEF)
    ef_data = ce_filtered[ce_filtered['itemid'] == 227008]
    ef_grouped = ef_data.groupby('hadm_id').agg(
        icu_LVEF_last=('valuenum', 'last')   # Last recorded LVEF for patient classification
    ).reset_index()
    
    final_df = final_df.merge(hr_grouped, on='hadm_id', how='left')
    final_df = final_df.merge(ef_grouped, on='hadm_id', how='left')
except Exception as e:
    print("Error processing chartevents:", e)


# ==========================================
# 9. Save final dataset
# ==========================================
final_df.to_csv('hf_guideline_driven_features.csv', index=False)

print(f"\nTotal clinical and significant features (including ICU): {final_df.shape[1]} columns")
print("Final file 'hf_guideline_driven_features.csv' created successfully!")