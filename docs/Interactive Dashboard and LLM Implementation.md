# Interactive Dashboard and LLM Implementation

This document describes the architecture, implementation logic, and data flow of the interactive dashboard and the AI-based Clinical Decision Support System. By combining the risk estimated by a logistic regression model with an LLM to add clinical reasoning capabilities, this dashboard provides a comprehensive tool for predicting and managing the 30-day readmission risk of heart failure patients based on 2022 AHA/ACC/HFSA Heart Failure Guideline Management guidelines.

## 1. Risk Prediction
The core analytical engine of this dashboard is built upon a logistic regression model and optimized parameters extracted from the training phase. To prevent processing delays, model artifacts—including regression weights, the scaler, and the optimal decision threshold—are loaded directly into cache memory upon dashboard startup.

**Feature Selection Logic:** Due to the limited volume of data used in this project and to prevent overfitting, feature selection phase and statistical testing identified only 5 features as statistically significant predictors for risk estimation. These features are:

•	Prescribe Beta Blocker during the stay

•	Hemoglobin at Admission

•	Potassium at Admission

•	Hemoglobin Change during the stay

•	Heart Rate at Discharge

Mathematical risk estimation in this system is performed exclusively using these 5 features. However, in the background, the system extracts several other features (such as diabetes status, blood pressure, renal status, and other medications) from the patient's record. These surplus features play no role in the logistic regression calculations; rather, their extraction is intended to be fed into the Large Language Model (LLM) so that the AI gains a deep, comprehensive, and multidimensional understanding of the patient's key indicators for providing guideline-based clinical recommendations.

### Three-Level Alert System and Safety Margin

In medicine, discharging a patient who actually requires further care (a false-negative error) carries life-threatening risks. Therefore, the model's output is not merely a binary zero-or-one decision; instead, the system utilizes a "safety margin" to create a caution zone.
If P is the risk probability calculated by the model and T is the optimal decision threshold, the dashboard classifies patients as follows:

•	**High-Risk Patient (Red Alert):** If P≥T, the patient is at high risk of requiring hospital readmission, and the system issues an immediate referral order for heart failure management programs and medication optimization.

•	**Borderline-Risk Patient (Yellow Alert):** If P falls within the range of T−5% to T, the patient is at medium (borderline) risk. This 5% range below the threshold serves as a safety margin. Although these patients are mathematically below the danger threshold, their proximity to the threshold requires closer monitoring and adjustment of discharge medications.

•	**Low-Risk Patient (Green Status):** If P<T−5%, the patient is at low risk of readmission, evaluated as clinically stable, and routine discharge programs are recommended.

## 2. Explainable AI - SHAP
A major barrier to the adoption of artificial intelligence systems by medical staff is their lack of transparency and "black box" nature. To build trust, this dashboard utilizes local interpretability techniques using the SHAP algorithm.
The system plots a waterfall chart for each selected record. This chart clearly illustrates how each of the 5 influential features steered the base value of the logistic regression model toward increasing risk (readmission) or decreasing risk. To facilitate clinical understanding, the system displays the patient's raw, actual (unnormalized) values next to the chart so the physician can understand the exact root cause of the calculated risk percentage.

## 3. Physiological Evaluation & Visual Analysis (Radar & Trend Analysis)
In addition to statistical risk estimation, the system is equipped with modules for clinical evaluation of biomarkers and vital signs.

**A) Radar Chart and Dynamic Normalization Formula:**

Simultaneously comprehending multiple laboratory parameters with completely different measurement units (such as sodium on a scale of 130 and NT-proBNP on a scale of thousands) at a glance is time-consuming. To solve this problem, all these values are mapped onto a standard geometric scale from 0 to 4.5. The system fetches the clinical reference range (lower and upper limits) for each test and normalizes the patient's values using the following equations:

$$Midpoint = \frac{Upper + Lower}{2}$$

$$Half\_Range = \frac{Upper - Lower}{2}$$

$$Normalized\_Value = 2.0 + \frac{Value - Midpoint}{Half\_Range}$$

In this geometric space, the number 2 represents the ideal value (the middle of the normal range), the number 1 represents the lower normal boundary, and the number 3 represents the upper normal boundary. The system draws a green polygon as the "safe zone" (between 1 and 3) and plots the patient's profile as a red band over it. Protrusion of the patient's profile vertices outside this green zone alerts the physician to hemodynamic instability or biomarker abnormalities.

**B) Longitudinal Trends of Biomarkers:**

To monitor the trajectory of a patient's recovery or deterioration, the dashboard fetches historical records of key indicators. Given the massive volume of laboratory data, to prevent memory overflow, the system processes data files in chunks (scanning 1 million rows per step) and plots time-series trends of indicators such as creatinine or potassium up to the moment of discharge as interactive charts.

## 4. RAG-Based Reasoning Engine and LLM Integration

The most advanced part of this system is the use of natural language processing to generate personalized, evidence-based clinical reports. To prevent "hallucination" errors in language models, a Retrieval-Augmented Generation (RAG) architecture is implemented as follows:

**A) Knowledge Base Creation and Vector Database:**

The system splits the text of the 2022 American Heart Association (AHA/ACC/HFSA) guidelines into 1,500-character text chunks (with 300-character overlap). Then, using the specialized `pritamdeka/S-PubMedBert-MS-MARCO model`, these texts are converted into semantic embeddings and stored in a Chroma vector database. Using a model trained on medical texts ensures that cardiovascular terminology is encoded with accurate semantic understanding.

**B) MMR Retrieval and Diversification Algorithm:**

The workflow in this section operates such that when a user requests a report generation, the system converts all patient information (including the 5 model features, other extracted clinical features, risk percentage, and alert status) into a structured query text.
This text is used to query the vector database to retrieve the most relevant guideline sections. Instead of simple similarity search, the Maximal Marginal Relevance (MMR) algorithm is utilized here. This algorithm first identifies 50 chunks relevant to the patient's condition, but ultimately extracts the 10 most relevant chunks out of those 50 that, in addition to semantic relevance to the query, exhibit maximum thematic diversity relative to one another, avoiding redundant sections. This mechanism ensures, for instance, that instead of receiving 10 duplicate texts regarding a single medication, the LLM simultaneously receives a diverse set of guidelines concerning drug titration, blood pressure control, and the management of the patient's underlying conditions.

**C) Strict Prompt Engineering:**

Patient information along with the extracted guideline texts are sent via a tightly controlled prompt to the powerful `llama-3.3-70b-versatile model`. The model is mandated to strictly limit its analysis to the guideline framework and provide output across exactly three sections: "Status Summary," "Guideline-Directed Medical Therapy Optimization," and "Discharge Plan." The key point is that the risk percentage calculated by the mathematical model and the patient's risk status are explicitly fed to the LLM as a clinical fact, ensuring that the AI's textual reasoning aligns precisely with the system's mathematical logic and alert level.

**D) Clinical Safety Checks:**

As a protective layer, the system reviews the vector database output prior to sending it to the LLM. If for any reason the database fails to find relevant guideline documentation for the patient's condition, the report generation process halts immediately, and a missing-documentation error is issued. This approach ensures that the AI never prescribes clinical recommendations based on its default knowledge, preventing potential harm to the patient.

## 5. UI Integration & System Simulation
The user interface, developed using the Streamlit framework, is designed to mirror the structure of an Electronic Health Record (EHR), allowing the user to call up patient information during a specific admission using patient ID and admission ID in the sidebar. 
The field placement logic is designed to be robust: if the user chooses not to fetch real patient data and simply intends to simulate a scenario, the system initializes empty fields with the mean value of that feature from the base dataset.
Finally, by injecting JavaScript code into the user interface, the system enables physicians to generate a comprehensive PDF file containing all analyses, charts, and the LLM reasoning report, attaching it directly to the patient's medical record.

