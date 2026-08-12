# Model Evaluation & Implementation

This section describes the training, evaluation, and implementation of the developed machine learning model. Given the limited sample size in the available dataset, this phase was conducted in two distinct stages: Stage 1, a rigorous evaluation of the model to assess its generalizability using subject-wise cross-validation; and Stage 2, training the final model on the entire dataset and packaging it for deployment in the Clinical Decision Support System.
The logic and mechanism of each stage are detailed below.

## 1. Model Evaluation and Validation (from `5_Fold_CV_ML.py`)

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

•	Aggregated Confusion Matrix: To visualize the overall distribution of correct and incorrect predictions.

# 2. Implementation and Deployment of the Final Model (from `Implement_ML.py`)

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

