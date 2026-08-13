# Feature Selection, ML model Evaluation and Implementation

This part outlines the methodology utilized to develop, validate, and deploy a machine learning model for predicting 30-day hospital readmissions among heart failure patients. The workflow is structured into three primary phases: First, stringent Data Preprocessing and Feature Selection, which employs K-Nearest Neighbors (KNN) imputation and statistically driven backward elimination to isolate the most clinically significant and interpretable variables. Second, rigorous Model Evaluation and Validation, featuring subject-wise cross-validation to prevent data leakage, alongside customized clinical thresholding and unbiased PPV calculations to ensure real-world reliability. Finally, Implementation and Deployment, wherein the finalized Logistic Regression model is trained on the entire dataset and packaged into a consolidated pipeline for seamless integration into a real-time Clinical Decision Support System.

## 1. Data Preprocessing & Feature Selection (from `feature_selection.py`)
Before evaluating and deploying the machine learning model, the initial pool of 37 guideline-driven clinical features undergoes a rigorous, multi-step preprocessing and feature selection pipeline. This phase ensures that the final model is not only statistically robust and computationally efficient but also highly interpretable for clinical decision-making.
The feature selection process is conducted through the following systematic steps:

## Data Quality Control and Imputation
To maintain data integrity, features with an excessive degree of missingness (greater than 70%) are initially removed from the dataset. For the remaining clinical variables, missing values are estimated using a K-Nearest Neighbors (KNN) imputer (with $k=5$). Algorithmically, when a specific data point is missing for a record, the system calculates the multidimensional distance across all available features to identify the 5 records with the most similar overall clinical profiles. The missing value is then replaced by the average value of that feature among these 'nearest neighbors'. Unlike simple mean imputation—which applies a single, population-wide average—the KNN approach preserves the underlying clinical relationships between variables, providing a highly personalized and realistic estimation of a patient's missing physiological or laboratory data.

## Variance and Collinearity Filtering
A Logistic Regression model is highly sensitive to redundant data and multicollinearity, which can distort model coefficients and mislead clinical interpretations. To prevent this:
**1.	Zero/Low Variance Removal:** A Variance Threshold (0.01) is applied to eliminate constant or near-constant features that provide no discriminatory power between patients.

**2.	High Linear Correlation Removal:** A pairwise correlation matrix is calculated for all variables. If two features exhibit a high linear correlation (absolute Pearson coefficient > 0.75), one of them is dropped. This ensures that each feature provided to the model brings unique, non-overlapping clinical information.

## Backward Elimination based on Statistical Significance
After scaling the data using a Z-score scaler, a statistically driven Backward Elimination method is utilized to finalize the feature subset. This approach ensures that only variables with a proven, independent statistical relationship to the target (30-day readmission) are retained.
The process is iteratively executed using the `statsmodels` library:

1.	A Logistic Regression model is fitted to the current set of features, and the P-Value for each feature is calculated.
2.	The feature with the highest P-Value is identified. If its P-Value exceeds the strict significance threshold of 0.05, it is considered statistically insignificant and is dropped from the dataset.
3.	The model is then re-fitted with the remaining features.
4.	This elimination loop continues strictly until all surviving features independently demonstrate strong statistical significance (P-Value $\le$ 0.05).

Table 1 presents the final statistically significant features retained after this iterative process, alongside their statistical metrics.

<p align="center">
  <b>Table 1: Identified significant features </b><br>
  <img src="https://github.com/user-attachments/assets/8f0f1e9c-032f-44d5-a566-de791f4fdc60" width="100%">
</p>

In clinical modeling, while the P-value confirms statistical significance, the Odds Ratio (OR) provides crucial clinical interpretability. To understand the OR correctly, it is essential to distinguish "odds" from "probability." Probability is the chance of an event occurring out of all possible outcomes, whereas odds represent the ratio of the probability that the event will occur to the probability that it will not occur (Event / Non-Event). The OR measures the multiplicative change in these odds for a one-unit increase in a given feature, assuming all other variables remain constant.

For instance, Potassium at Admission has an OR of 1.8665. This means that for every one-unit increase in this standardized feature, the odds of readmission increase by approximately 86.6%, which is entirely distinct from increasing the overall probability by that amount. Generally, an OR greater than 1 indicates that the feature amplifies the risk of the event, whereas an OR less than 1 (e.g., Hemoglobin at Admission) signifies an inverse relationship, meaning an increase in that feature lowers the odds of readmission.
	
Finally, these significant features used to train and validate Logistic Regression model as a risk predictor.


## 2. Model Evaluation and Validation (from `5_Fold_CV_ML.py`)

The objective of this section is to evaluate the model's performance on unseen data and prevent common pitfalls in medical modeling, such as data leakage and overfitting. The chosen machine learning algorithm for this system is Logistic Regression with an $L2$ penalty, which provides a good balance of accuracy and high interpretability for clinical variables.

## Subject-wise Cross-Validation (Managing Data Leakage)

A fundamental challenge with this data is that a single patient may have multiple admissions. If the data is randomly split into training and test sets, one admission of a patient might end up in the training set while another admission of the same patient falls into the test set. This causes the model to memorize the patient's data, yielding falsely inflated performance. To resolve this, a Subject-wise 5-fold cross-validation approach is utilized. The data is partitioned into 5 fold based on the patient ID (`subject_id`), ensuring that all records of a specific patient are entirely allocated to either the training or the test set.
Furthermore, to prevent statistical data leakage, feature scaling (StandardScaler) is performed independently within each fold. That is, the scaler is fitted solely on the training data of that specific fold and then applied to the test data.

## Clinical Thresholding

In clinical applications, the cost of misclassification is not symmetric. For readmission prediction, failing to identify a patient who genuinely requires readmission (False Negative) is far more dangerous than issuing a false alarm for a healthy patient (False Positive). Therefore, instead of using the default 0.5 threshold, the system optimizes the decision threshold based on a target sensitivity of 80%. This reduces the False Negative (FN) rate, minimizing the chance that a high-risk patient is incorrectly classified as low-risk. In each training iteration, the model identifies a threshold in the training data that correctly identifies at least 80% of the high-risk patients, and then applies this threshold to predict outcomes on the test data. This process is repeated 5 times, and ultimately, the evaluation metrics are reported for each fold along with their mean and standard deviation.

## Evaluation Metrics Overview
To ensure a comprehensive understanding of the model's performance, the following clinical and statistical metrics are utilized:

•	Sensitivity (Recall / True Positive Rate): The percentage of actual readmitted patients correctly identified by the system as high-risk.

•	Specificity (True Negative Rate): The percentage of non-readmitted patients correctly identified as low-risk.

•	PPV (Positive Predictive Value / Precision): The probability that a patient flagged by the system as high-risk will actually be readmitted.

•	NPV (Negative Predictive Value): The probability that a patient flagged as low-risk will safely stay out of the hospital.

•	AUROC (Area Under the ROC Curve): Measures the model's overall capability to distinguish between the two classes.

•	AUPRC (Area Under the Precision-Recall Curve): A highly reliable metric for imbalanced data that summarizes the precision-recall trade-off.

•	Brier Score: Evaluates the accuracy of the predicted probabilities (a lower score indicates better calibration).

## Unbiased Positive Predictive Value (Unbiased PPV)
Although the class ratio in this system (readmitted vs. non-readmitted patients — ~24%) is somewhat close to the actual prevalence of this event (~18.4% in the US), discrepancies between a dataset's internal prevalence and the real-world prevalence generally cause the standard PPV to be miscalculated. In this system, considering the actual heart failure readmission prevalence in the US population (18.4%), the PPV is calculated in an unbiased manner using Bayes' theorem as follows:

$$\text{Unbiased PPV} = \frac{\text{Sensitivity} \times \text{Prevalence}}{(\text{Sensitivity} \times \text{Prevalence}) + ((1 - \text{Specificity}) \times (1 - \text{Prevalence}))}$$

This formula assures the clinician of the true real-world probability of the risk occurring when the system triggers an alert.


## Statistical Evaluation and 95% Confidence Interval 

To ensure that the model's evaluation results are not merely dependent on one specific data split, the predictions across all 5 folds are aggregated. Then, using a Bootstrapping technique with 1000 iterations, the 95% Confidence Interval (95% CI) is calculated for all metrics. This statistically validates the model's certainty.

In addition to numerical metrics, the system generates four key charts for visual evaluation:

•	ROC Curve: To evaluate the model's ability to separate the classes.

•	Precision-Recall Curve: A much more accurate metric than ROC for imbalanced datasets.

•	Calibration Curve: Accompanied by a confidence interval band, showing how well the model's output probabilities align with reality.

## Cross-Validation Results and Clinical Trade-offs
As detailed in the table 2 and visually represented in the plotted curves in figure 1, the model demonstrates an acceptable and robust performance, particularly when considering the limited sample size and the inherently imbalanced nature of hospital readmission data. Because standard metrics like AUROC (which achieved a mean of 0.716) can present an overly optimistic view in imbalanced scenarios, the model's true discriminative ability is more accurately evaluated using the Area Under the Precision-Recall Curve (AUPRC). For this specific dataset, the baseline performance of a random classifier on the AUPRC is strictly equal to the positive class prevalence, which averages 0.245 across the folds. As shown in the table 2, the model achieves a mean AUPRC of 0.520, which is more than double the random baseline, proving its genuine effectiveness in capturing high-risk patterns rather than relying on majority-class guessing. Furthermore, the accuracy of the estimated risk probabilities is validated by a impressively low mean Brier score of 0.168. As illustrated in the Calibration Curve, the aggregated predictions closely track the perfectly calibrated line and remain well within the 95% bootstrap confidence interval. This robust risk calibration is a critical requirement for a Clinical Decision Support System, ensuring that if, for instance, a logistic regression predicts a 60% readmission probability, it genuinely translates to a 60% real-world risk, allowing physicians to confidently act upon the system's quantitative alerts.

While the evaluation table reveals a comparatively lower Positive Predictive Value (mean PPV of 0.342) and Unbiased PPV, this behavior is not a computational flaw but a deliberate, mathematically expected outcome of optimizing the decision threshold for high sensitivity. In the clinical management of heart failure, the cost of misclassification is highly asymmetric; the primary objective is to aggressively minimize False Negatives (FN), as discharging a high-risk patient prematurely carries severe, potentially fatal consequences. Conversely, the clinical and financial cost of a False Positive (FP) is relatively minimal, typically resulting only in medication optimization, extended observation, or scheduling an earlier outpatient follow-up. By embracing this strategic trade-off, the model achieves a strong Negative Predictive Value (NPV) averaging 0.810. From a practical perspective, this high NPV demonstrates that when the system classifies a patient as low-risk, healthcare providers can be 81% confident that the patient will safely stay out of the hospital. However, the system intentionally sacrifices a degree of precision (PPV) to strictly suppress the False Negative rate. 

It should be noted that all plots presented in Figure 1 are generated using the aggregated predictions from all five test sets, thereby representing the model's generalized performance across the entire cohort. 

<p align="center">
  <b>Table 2: Evaluation metrics derived from 5-fold subject-wise cross validation </b><br>
  <img src="https://github.com/user-attachments/assets/d83ecaf6-a9cf-498a-ad4e-d1eac5b4c0a9" width="100%">
</p>

<div align="center">
  <br>
  <img src="https://github.com/user-attachments/assets/ea3b7124-1eb7-425f-b483-d29ebca110d2" width="100%">
  
<p align="left">
    <i><strong>Fig. 1: evaluation plot of the model based on aggregated predictions from all five test sets. (a) Receiver Operating Characteristic (ROC) curve. (b) Precision-Recall Curve (PRC). (c) Calibration curve with 95% bootstrap confidence interval.</strong></i>
  </p>

</div>


# 3. Implementation and Deployment of the Final Model (from `Implement_ML.py`)

After validating the machine learning model and identifying the hyperparameters, the final model is built for deployment in a production environment.
Utilizing 100% of the Data Capacity

Since the sample size in the available dataset is relatively small, setting aside a portion of the data for testing the deployment model leads to unnecessary information loss. For this reason, the final model is trained on the entire available dataset to achieve maximum inference power. In this script, no Train/Test split is performed. Both the scaler (StandardScaler) and the model (LogisticRegression) are fitted on all patient records.

## Implemented final Threshold

Following model training, prediction probabilities are calculated for the entire dataset. Then, by calulating the ROC curve on all data, a cut-off threshold is extracted that minimizes the False Negative error by achieving a sensitivity of over 80%. This threshold will serve as the final decision boundary in the dashboard.

## Deployment Package
A machine learning model alone is insufficient for predicting a new patient's status. The new patient's data must be standardized using the exact same scale as the training data and evaluated using the identical threshold. At the end of this pipeline, all vital components are compiled into a single dictionary:

•	model: The trained Logistic Regression object.

•	scaler: The scaler containing the mean and standard deviation of the clinical features.

•	threshold: The calculated decision threshold for >80% sensitivity.

•	features: The exact order and names of the clinical columns/features.

This package is saved using the joblib library into a file named hf_deployment_package.pkl. The final software or Interactive Dashboard will have access to the complete prediction system simply by loading this single file, allowing it to receive new patient data and calculate their readmission risk in real-time.

