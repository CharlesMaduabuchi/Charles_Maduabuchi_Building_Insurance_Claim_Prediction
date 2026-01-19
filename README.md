# Building Insurance Claim Prediction#
##📌 Project Goal
To build a predictive model that determines the probability of a building having at least one insurance claim. This assists in risk assessment and premium optimization for insurance providers.

## 🛠️ Key Technical Steps
Data Cleaning: Handled missing values in Building Dimension and Date_of_Occupancy. Sanitized the NumberOfWindows column by resolving non-numeric entries (. and >=10).

Feature Engineering: Created a Building_Age feature (Year of Observation - Date of Occupancy) to better capture structural risk.

Preprocessing: Implemented One-Hot Encoding for categorical variables and Standard Scaling for numerical features.

Class Imbalance: Applied class_weight='balanced' to improve the detection of claims in an imbalanced dataset.

## 📈 Model Performance#
I evaluated two models to compare baseline linear performance against non-linear complexity:

Logistic Regression: Provided a baseline with high interpretability.

XGBoost/Random Forest: Achieved a higher ROC-AUC (~0.74), successfully capturing complex interactions between building size and location.

## 📂 Repository Structure
Train_data.csv: Historical building and claim data.

Insurance_Claim_Analysis.ipynb: Full pipeline from EDA to Model Evaluation.

model.pkl: The final trained predictive model.

Role: Lead Data Analyst

Tools: Python, Scikit-Learn, XGBoost, Pandas, Seaborn.
