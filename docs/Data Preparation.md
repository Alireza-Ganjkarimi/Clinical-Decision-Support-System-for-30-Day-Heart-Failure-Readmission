# Data Preparation & Feature Engineering

In this phase of developing the Clinical Decision Support System for heart failure patients, the primary objective is to transform raw and scattered hospital data (based on the [demo version of MIMIC-IV database](https://www.kaggle.com/datasets/montassarba/mimic-iv-clinical-database-demo-2-2/data) ) into a structured and integrated feature matrix. The features extracted in this phase are selected in full alignment with established clinical guidelines ([2022 AHA/ACC/HFSA Heart Failure Management Guidelines](https://www.ahajournals.org/doi/10.1161/CIR.0000000000001063)), ensuring that the predictive model makes decisions grounded in medical evidence.

This phase consists of two essential steps: 1. Establishing the base cohort and defining the target variable, and 2. Extracting clinical features.

## Step 1: Cohort Establishment and Target Variable Definition

In this stage, implemented in `merge.py`, target patients are filtered, and the final label for machine learning model training is generated. This process involves the following steps:

### Identifying Heart Failure Patients

First, the system examines the diagnosis file (diagnoses_icd.csv). Using standard international disease codes (ICD-9 and ICD-10), patients with a history of heart failure (such as codes in the 428 group, I50, and other specific heart failure codes) are identified. Subsequently, from the admission information file (admissions.csv), only records corresponding to these patients are extracted so that the cohort exclusively includes heart failure patients.

### Temporal Tracking and Time-to-Readmission Calculation

To predict the probability of patient readmission, we need to maintain the chronological sequence of records. All hospital stay records for each patient are sorted by admission time. The system then compares the discharge time of the current stay with the admission time of the subsequent stay to calculate the number of days elapsed until readmission.

### Defining the Target Variable (30-Day Readmission)

One of the most critical indicators for evaluating heart failure treatment quality is preventing readmission within less than a month. Accordingly, the target variable named `readmission_30d` is defined as a binary variable (zero and one):

**Value 1 (Positive class):** If the time interval between the patient's discharge and their subsequent admission is between 0 and 30 days, indicating early readmission.

**Value 0 (Negative class):** If the patient is readmitted after 30 days or has no readmission at all.

### Applying Exclusion Criteria

Finally, a crucial logical filter is applied based on discharge status. At this stage, stay records where the patient expired in the hospital  are excluded from the dataset. By doing so, a patient's entire history is not deleted; rather, if a patient experienced multiple successful stays and discharges prior to expiration, their prior stay data is fully preserved in the dataset, and only the specific record of the final stay leading to death is filtered out. The clinical rationale for this is clear: in a stay resulting in death, the patient does not return home to become a candidate for 30-day readmission risk assessment, and including these specific rows would introduce computational error into the predictive model. The output of this stage is a base file of all valid admissions paired with the prediction label.

## Step 2: Clinical Feature Extraction

In this stage, developed in `making_feature_matrix.py`, the system accesses various database tables to attach physiological and clinical features of each stay to the base file. The logic for extracting each category of information is as follows:

### Extracting Biomarkers and Laboratory Indicators

Laboratory information plays a vital role in diagnosing heart failure deterioration. By querying `labevents.csv`, the system tracks six key factors emphasized in clinical guidelines: NT-proBNP, creatinine, blood urea nitrogen (BUN), sodium, potassium, and hemoglobin. An important point in this section is that instead of extracting a simple average, the system creates three variables for each factor over the course of a hospital stay: the initial value (First), the final value prior to discharge (Last), and the difference between the two (Delta). Extracting the Delta variable helps the model effectively understand the patient's recovery or deterioration trajectory throughout the treatment course.

### Blood Pressure Monitoring from Outpatient Records (OMR)

The system extracts blood pressure data from `omr.csv`. Since this data is recorded as text strings (e.g., "120/80"), the system first parses them into two  variables: systolic blood pressure (SBP) and diastolic blood pressure (DBP). Then, by precisely matching the blood pressure recording date with the patient's admission-to-discharge dates, it ensures that only data relevant to that specific hospital stay is included. Finally, statistical features including the minimum, maximum, first, last , and change (delta) values of systolic blood pressure during the stay, along with the last diastolic pressure are extracted.


### Extracting Comorbidities

Heart failure patients frequently suffer from comorbidities that increase readmission risk. Using ICD codes from `diagnoses_icd.csv`, the system searches each patient's record for four major conditions: atrial fibrillation (AFib), diabetes, hypertension, and chronic kidney disease (CKD). The presence or absence of each condition is added to the patient profile as binary variables.

### Guideline-Directed Medical Therapy (GDMT) Tracking

Taking the correct medications has a direct impact on reducing readmission rates. In this section, the prescription file (`prescriptions.csv`) is reviewed to determine whether the patient received medications recommended in heart failure protocols during their stay. These medications are categorized into four primary groups:

1- Beta-blockers (such as metoprolol and carvedilol)

2- ACE inhibitors / ARBs / ARNIs (such as lisinopril and valsartan)

3- Mineralocorticoid receptor antagonists or MRAs (such as spironolactone)

4- Loop diuretics for fluid management (such as furosemide

By searching for keyword names of these drugs, the system records the receipt or non-receipt of each medication group as an independent feature.

### Intensive Care Unit Vitals & Congestion Indicators

For patients with more severe conditions admitted to the intensive care unit, vital data is extracted from two sources:

**Fluid Congestion Status:** By examining `outputevents.csv`, the total urine output or fluid balance of the patient during the ICU stay is calculated, which is a vital indicator for evaluating the patient's response to diuretic medications.

**Heart Rate and Left Ventricular Ejection Fraction (LVEF):** Values related to left ventricular ejection fraction (to determine the type of heart failure) and heart rate values (including minimum heart rate to detect bradycardia, maximum heart rate for tachycardia, and the last recorded heart rate) are extracted from `chartevents.csv`.

### Final Output of the Preparation Phase

At the end of this phase, all processed data from various tables are merged based on the hospital admission ID (`hadm_id`). The result of this complex process is the generation of `hf_guideline_driven_features.csv`—a rich data matrix containing a valid target variable and a collection of physiological, laboratory, and pharmacological features fully prepared for training artificial intelligence algorithms and risk analysis.
