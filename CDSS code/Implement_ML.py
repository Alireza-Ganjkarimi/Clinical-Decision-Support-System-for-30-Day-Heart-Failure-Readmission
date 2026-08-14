import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (roc_curve)

print("\n--- Training Final Production Model on 100% of Data ---")

df = pd.read_csv('hf_model_ready_data_pvalue_filtered.csv')
meta_cols = ['subject_id', 'hadm_id', 'admittime', 'dischtime', 'readmission_30d']

y = df['readmission_30d']
groups = df['subject_id']

X = df.drop(columns=meta_cols)

target_sensitivity=0.80 # Clinical goal: At least achieve 80% sensitivity to minimize False Negatives (FN - patients who are readmitted but incorrectly classified as not readmitted).

# 1. Create and fit the Scaler on the entire dataset
final_scaler = StandardScaler()
X_all_scaled = final_scaler.fit_transform(X)

# 2. Train the final model on the entire dataset
final_model = LogisticRegression(
    penalty='l2', 
    C=1,  
    solver='liblinear',
    random_state=42
)


final_model.fit(X_all_scaled, y)

# 3. Calculate the final threshold on the entire dataset to achieve 80% sensitivity
y_pred_prob_all = final_model.predict_proba(X_all_scaled)[:, 1]
fpr_all, tpr_all, thresholds_all = roc_curve(y, y_pred_prob_all)

valid_indices_all = np.where(tpr_all >= target_sensitivity)[0]
final_threshold = thresholds_all[valid_indices_all[0]] if len(valid_indices_all) > 0 else 0.5

fpr_at_threshold = fpr_all[valid_indices_all[0]]
achieved_sensitivity = tpr_all[valid_indices_all[0]]
specificity_at_threshold = 1 - fpr_at_threshold

print(f"Final Production Threshold (Target Sensitivity > 80%): {final_threshold:.4f}")
print(f"Achieved Sensitivity (TPR): {achieved_sensitivity:.4f}")
print(f"Achieved Specificity (TNR): {specificity_at_threshold:.4f}")


# 4. Save all three components in a single file for use in the interactive dashboard
deployment_package = {
    'model': final_model,
    'scaler': final_scaler,
    'threshold': final_threshold,
    'features': X.columns.tolist() # Save column names for building the user interface
}

joblib.dump(deployment_package, 'hf_deployment_package.pkl')
print("Final package successfully saved to 'hf_deployment_package.pkl'.")