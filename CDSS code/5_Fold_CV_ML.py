import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             confusion_matrix, accuracy_score, f1_score,
                             roc_curve, precision_recall_curve, auc,
                             ConfusionMatrixDisplay) # Added ConfusionMatrixDisplay here
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

# 1. Load Data
df = pd.read_csv('hf_model_ready_data_pvalue_filtered.csv')
meta_cols = ['subject_id', 'hadm_id', 'admittime', 'dischtime', 'readmission_30d']
y = df['readmission_30d']
groups = df['subject_id']
X = df.drop(columns=meta_cols)

# --- Clinical Tuning Variables ---
true_prevalence = 0.184  # 30-day readmission prevalence rate for heart failure patients, based on the US population. Used to calculate unbiased PPV.
target_sensitivity = 0.80  # Clinical goal: At least achieve 80% sensitivity to minimize False Negatives (FN - patients who are readmitted but incorrectly classified as not readmitted).
print(f"Assumed True Prevalence: {true_prevalence:.3f}")
print(f"Target Sensitivity for Thresholding: {target_sensitivity:.2f}")
# ------------------------------

# 2. Define the Standardized Model
scaler = StandardScaler()
model = LogisticRegression(
    penalty='l2', 
    C=1,  
    solver='liblinear',
    random_state=42
)

# 3. Patient-based Cross-Validation
gkf = GroupKFold(n_splits=5)

# Added 'Prevalence' to the metrics dictionary
metrics = {
    'AUROC': [], 'AUPRC': [], 'Brier': [],
    'SE': [], 'SP': [], 'PPV': [], 'Unbiased_PPV': [], 'NPV': [], 'ACC': [], 'F1': [], 'Prevalence': []
}

y_true_all = []
y_prob_all = []
y_pred_all = [] # Added to store predicted classes across all folds for the aggregated confusion matrix

print("\n--- Performance Metrics Per Fold ---")
fold_idx = 1

for train_idx, test_idx in gkf.split(X, y, groups=groups):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    
    # Standardization (preventing data leakage)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Extract the best threshold on the training data
    y_pred_prob_train = model.predict_proba(X_train_scaled)[:, 1]
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_pred_prob_train)
    
    valid_indices = np.where(tpr_train >= target_sensitivity)[0]
    if len(valid_indices) > 0:
        best_idx = valid_indices[0]
        best_thresh_train = thresholds_train[best_idx]
    else:
        best_thresh_train = 0.5
    
    # Predict probabilities on the test data
    y_pred_prob_test = model.predict_proba(X_test_scaled)[:, 1]
    
    y_true_all.extend(y_test)
    y_prob_all.extend(y_pred_prob_test)
    
    if len(np.unique(y_test)) > 1:
        fold_auroc = roc_auc_score(y_test, y_pred_prob_test)
        fold_auprc = average_precision_score(y_test, y_pred_prob_test)
        fold_brier = brier_score_loss(y_test, y_pred_prob_test)
        
        # Apply the extracted threshold
        y_pred_class = (y_pred_prob_test >= best_thresh_train).astype(int)
        y_pred_all.extend(y_pred_class) # Save predicted classes for aggregated evaluation
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_class).ravel()
        
        fold_se = tp / (tp + fn) if (tp + fn) > 0 else 0
        fold_sp = tn / (tn + fp) if (tn + fp) > 0 else 0
        fold_ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        fold_npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        fold_acc = accuracy_score(y_test, y_pred_class)
        fold_f1 = f1_score(y_test, y_pred_class)
        
        # Calculate Class 1 prevalence for this fold
        fold_prevalence = np.mean(y_test) 
        
        # Calculate Unbiased PPV
        ppv_numerator = fold_se * true_prevalence
        ppv_denominator = ppv_numerator + ((1 - fold_sp) * (1 - true_prevalence))
        fold_unbiased_ppv = ppv_numerator / ppv_denominator if ppv_denominator > 0 else 0
        
        metrics['AUROC'].append(fold_auroc)
        metrics['AUPRC'].append(fold_auprc)
        metrics['Brier'].append(fold_brier)
        metrics['SE'].append(fold_se)
        metrics['SP'].append(fold_sp)
        metrics['PPV'].append(fold_ppv)
        metrics['Unbiased_PPV'].append(fold_unbiased_ppv) 
        metrics['NPV'].append(fold_npv)
        metrics['ACC'].append(fold_acc)
        metrics['F1'].append(fold_f1)
        metrics['Prevalence'].append(fold_prevalence) # Append fold prevalence
        
        # Print all metrics for this fold, now including Prev (Prevalence)
        print(f"Fold {fold_idx:2d} | Thresh: {best_thresh_train:.3f} | "
              f"AUROC: {fold_auroc:.3f} | AUPRC: {fold_auprc:.3f} | Brier: {fold_brier:.3f} | "
              f"SE: {fold_se:.3f} | SP: {fold_sp:.3f} | PPV: {fold_ppv:.3f} | "
              f"uPPV: {fold_unbiased_ppv:.3f} | NPV: {fold_npv:.3f} | ACC: {fold_acc:.3f} | F1: {fold_f1:.3f} | Prev: {fold_prevalence:.3f}")
    else:
        print(f"Fold {fold_idx:2d} | Skipped (Only one class present in test set)")
        
    fold_idx += 1

# ========================================================
# Bootstrap 95% CI Calculation on Aggregated Data
# ========================================================
y_true_np = np.array(y_true_all)
y_prob_np = np.array(y_prob_all)
y_pred_np = np.array(y_pred_all)

n_bootstraps_metrics = 1000
rng = np.random.RandomState(42)
bootstrap_results = {k: [] for k in metrics.keys()}

for _ in range(n_bootstraps_metrics):
    idx = rng.choice(len(y_true_np), len(y_true_np), replace=True)
    y_true_b = y_true_np[idx]
    y_prob_b = y_prob_np[idx]
    y_pred_b = y_pred_np[idx]
    
    # Skip if only one class is present in the bootstrap sample
    if len(np.unique(y_true_b)) < 2:
        continue
        
    bootstrap_results['AUROC'].append(roc_auc_score(y_true_b, y_prob_b))
    bootstrap_results['AUPRC'].append(average_precision_score(y_true_b, y_prob_b))
    bootstrap_results['Brier'].append(brier_score_loss(y_true_b, y_prob_b))
    
    tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_true_b, y_pred_b).ravel()
    
    se_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0
    sp_b = tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0
    ppv_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0
    npv_b = tn_b / (tn_b + fn_b) if (tn_b + fn_b) > 0 else 0
    
    ppv_num = se_b * true_prevalence
    ppv_den = ppv_num + ((1 - sp_b) * (1 - true_prevalence))
    u_ppv_b = ppv_num / ppv_den if ppv_den > 0 else 0
    
    bootstrap_results['SE'].append(se_b)
    bootstrap_results['SP'].append(sp_b)
    bootstrap_results['PPV'].append(ppv_b)
    bootstrap_results['Unbiased_PPV'].append(u_ppv_b)
    bootstrap_results['NPV'].append(npv_b)
    bootstrap_results['ACC'].append(accuracy_score(y_true_b, y_pred_b))
    bootstrap_results['F1'].append(f1_score(y_true_b, y_pred_b))
    bootstrap_results['Prevalence'].append(np.mean(y_true_b)) # Append bootstrap prevalence

# 4. Report Final Results (Mean, Variance, and Aggregated 95% CI)
print("\n--- Final Aggregated Metrics (Mean ± Variance) with 95% CI ---")
for metric_name, values in metrics.items():
    if len(values) > 0:
        mean_val = np.mean(values)
        var_val = np.var(values)
        if len(bootstrap_results[metric_name]) > 0:
            lower_ci = np.percentile(bootstrap_results[metric_name], 2.5)
            upper_ci = np.percentile(bootstrap_results[metric_name], 97.5)
            print(f"{metric_name:<12}: {mean_val:.3f} (± {var_val:.4f}) | 95% CI: [{lower_ci:.3f} - {upper_ci:.3f}]")
        else:
            print(f"{metric_name:<12}: {mean_val:.3f} (± {var_val:.4f})")

# ========================================================
# 5. Plot charts based on aggregated predictions
# ========================================================
y_true_all = np.array(y_true_all)
y_prob_all = np.array(y_prob_all)

# --- Chart 1: ROC Curve ---
plt.figure(figsize=(7, 6))
fpr, tpr, _ = roc_curve(y_true_all, y_prob_all)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

# --- Chart 2: Precision-Recall Curve (PRC) ---
plt.figure(figsize=(7, 6))
precision, recall, _ = precision_recall_curve(y_true_all, y_prob_all)
pr_auc = average_precision_score(y_true_all, y_prob_all)
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUPRC = {pr_auc:.3f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall (Sensitivity)')
plt.ylabel('Precision (PPV)')
plt.title('Precision-Recall Curve (PRC)')
plt.legend(loc="lower left")
plt.grid(alpha=0.3)
plt.show()

# --- Chart 3: Calibration Curve with Confidence Interval ---
plt.figure(figsize=(7, 6))
prob_true, prob_pred = calibration_curve(y_true_all, y_prob_all, n_bins=3 , strategy='quantile')

# Calculate 95% Confidence Interval using Bootstrapping
n_bootstraps = 1000
interp_trues = []
eval_preds = np.linspace(0, 1, 100) # Fixed points for mean interpolation

for _ in range(n_bootstraps):
    idx = np.random.choice(len(y_true_all), len(y_true_all), replace=True)
    y_true_b, y_prob_b = y_true_all[idx], y_prob_all[idx]
    
    # Ignore samples that contain only one class
    if len(np.unique(y_true_b)) < 2:
        continue
        
    p_true_b, p_pred_b = calibration_curve(y_true_b, y_prob_b, n_bins=3 , strategy='quantile')
    
    if len(p_pred_b) > 1:
        # Interpolate over fixed points to allow calculating percentiles
        interp_true = np.interp(eval_preds, p_pred_b, p_true_b)
        interp_trues.append(interp_true)

# Plot the confidence interval band
if interp_trues:
    lower_ci = np.percentile(interp_trues, 2.5, axis=0)
    upper_ci = np.percentile(interp_trues, 97.5, axis=0)
    plt.fill_between(eval_preds, lower_ci, upper_ci, color='dodgerblue', alpha=0.3, label='95% CI (Bootstrap)')

plt.plot(prob_pred, prob_true, 's-', color='dodgerblue', lw=2, label='Model Calibration')
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfectly Calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives (True Probability)')
plt.title('Calibration Curve with 95% Confidence Interval')
plt.legend(loc='best')
plt.grid(alpha=0.3)
plt.show()

# --- Chart 4: Aggregated Confusion Matrix ---
plt.figure(figsize=(6, 5))
cm_aggregated = confusion_matrix(y_true_all, y_pred_all)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_aggregated, display_labels=['Class 0', 'Class 1'])
disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
plt.title('Aggregated Confusion Matrix (5 Folds)')
plt.show()