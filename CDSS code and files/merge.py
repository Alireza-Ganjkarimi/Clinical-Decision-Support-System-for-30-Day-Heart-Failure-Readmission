import pandas as pd
import numpy as np

# 1. Load data
diagnoses = pd.read_csv('diagnoses_icd.csv')
admissions = pd.read_csv('admissions.csv')

# 2. Convert date columns to Datetime format
admissions['admittime'] = pd.to_datetime(admissions['admittime'])
admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])

# 3. Find chronic heart failure patients
diagnoses['icd_code'] = diagnoses['icd_code'].astype(str)
specific_hf_codes = [
    '40201', '40211', '40291', '40401', '40403', '40411', '40413', '40491', '40493', '39891', # ICD-9
    'I110', 'I130', 'I132', 'I0981' # ICD-10
]
hf_condition = (
    diagnoses['icd_code'].str.startswith(('428', 'I50')) | 
    diagnoses['icd_code'].isin(specific_hf_codes)
)
hf_patients = diagnoses[hf_condition]['subject_id'].unique()

# 4. Filter admissions table for heart failure patients 
hf_admissions = admissions[admissions['subject_id'].isin(hf_patients)].copy()

# 5. Sort admissions by patient and admission time to preserve the chronological timeline
hf_admissions = hf_admissions.sort_values(by=['subject_id', 'admittime'])

# 6. Calculate the time to the next admission (Readmission) while keeping all records
hf_admissions['next_admittime'] = hf_admissions.groupby('subject_id')['admittime'].shift(-1)
hf_admissions['days_to_next_admission'] = (hf_admissions['next_admittime'] - hf_admissions['dischtime']).dt.days

# 7. Create target variable label
hf_admissions['readmission_30d'] = np.where(
    (hf_admissions['days_to_next_admission'] >= 0) & (hf_admissions['days_to_next_admission'] <= 30), 
    1, 
    0
)

# Select key and important columns for saving (remove extra columns)
final_columns = ['subject_id', 'hadm_id', 'admittime', 'dischtime', 'readmission_30d']
hf_base_cohort = hf_admissions[final_columns].copy()

# Save as a new CSV file on your system
hf_base_cohort.to_csv('hf_base_cohort.csv', index=False)

print("\nFile 'hf_base_cohort.csv' successfully saved!")

# 8. remove the records where the patient died 
# (Because a deceased patient does not go home to be monitored remotely)
hf_admissions = hf_admissions[hf_admissions['hospital_expire_flag'] == 0]

# Display results
print("Target Variable Distribution:")
print(hf_admissions['readmission_30d'].value_counts())

display_columns = ['subject_id', 'hadm_id', 'hospital_expire_flag', 'days_to_next_admission', 'readmission_30d']
print("\nSample Data:")
print(hf_admissions[display_columns].head(10))