import pandas as pd
try:
    from pandas.util._exceptions import find_stack_level
except ImportError:
    import warnings
    def find_stack_level():
        return 2
    import pandas.util._exceptions as pd_exceptions
    pd_exceptions.find_stack_level = find_stack_level
    
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
import statsmodels.api as sm
warnings.filterwarnings('ignore')

# 1. Load the 37 guideline-driven features data
df = pd.read_csv('hf_guideline_driven_features.csv')

meta_cols = ['subject_id', 'hadm_id', 'admittime', 'dischtime', 'readmission_30d']
y = df['readmission_30d']
groups = df['subject_id']
X = df.drop(columns=meta_cols)

print(f"Number of initial features (without metadata): {X.shape[1]}")

# Remove columns with over 70% missing values."

missing_threshold = 0.7 

X = X.loc[:, X.isnull().mean() < missing_threshold]
print(f"Number of features after dropping columns with high missing data: {X.shape[1]}")

# 2. Impute missing values using KNN
imputer = KNNImputer(n_neighbors=5)
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

# 3. Remove zero variances (constant values)
selector = VarianceThreshold(threshold=0.01)
selector.fit(X_imputed)
X_var = X_imputed.loc[:, selector.get_support()]

# 4. Remove high linear correlations
corr_matrix = X_var.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.75)]
X_uncorr = X_var.drop(columns=to_drop)

print(f"Number of features after removing zero variance and high correlation: {X_uncorr.shape[1]}")

# 5. Standardize the data
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_uncorr), columns=X_uncorr.columns, index=X_uncorr.index)

# 6. Feature selection using Backward Elimination based on P-Value
print("\n--- Start of Backward Elimination ---")
selected_features = list(X_scaled.columns)
p_value_threshold = 0.05
eliminated_history = []

while len(selected_features) > 0:
    # Build model with current features
    X_sm = sm.add_constant(X_scaled[selected_features])
    
    # Use Logit model in statsmodels
    logit_model = sm.Logit(y, X_sm)
    try:
        result = logit_model.fit(disp=False)
    except Exception as e:
        print("Error in model convergence:", e)
        break
        
    # Extract P-Values (excluding the intercept or const)
    p_values = result.pvalues.drop('const')
    
    # Find the highest P-Value
    max_p_value = p_values.max()
    max_p_feature = p_values.idxmax()
    
    if max_p_value > p_value_threshold:
        print(f"Dropping feature: '{max_p_feature}' with P-Value = {max_p_value:.4f}")
        selected_features.remove(max_p_feature)
        eliminated_history.append((max_p_feature, max_p_value))
    else:
        # Break the loop if no P-Value is above 0.05
        print("\nAll remaining features have a P-Value <= 0.05.")
        break

print(f"\nNumber of final statistically confirmed features: {len(selected_features)}")

# 7. Calculate Odds Ratio and final table with the confirmed model
X_final_sm = sm.add_constant(X_scaled[selected_features])
final_result = sm.Logit(y, X_final_sm).fit(disp=False)

# Extract final coefficients and P-Values
final_p_values = final_result.pvalues.drop('const')
final_coefs = final_result.params.drop('const')

p_val_df = pd.DataFrame({
    'Feature': selected_features,
    'Coefficient': final_coefs.values,
    'Odds_Ratio': np.exp(final_coefs.values),
    'P_Value': final_p_values.values
}).sort_values(by='P_Value', ascending=True)

print("\n--- Table of statistical values and P-Values for final features ---")
print(p_val_df.to_string(index=False))

# 8. Save the final dataset
final_selected_df = pd.concat([df[meta_cols], X_uncorr[selected_features]], axis=1)
final_selected_df.to_csv('hf_model_ready_data_pvalue_filtered.csv', index=False)
print("\nThe final dataset based on statistical significance was saved in 'hf_model_ready_data_pvalue_filtered.csv'.")