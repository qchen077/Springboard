# Bank Marketing Prediction (Imbalanced Classification with SMOTE) – Capstone Project 2

## 📌 Overview
This project analyzes a direct marketing campaign conducted by a Portuguese banking institution.  
The goal is to predict whether a customer will subscribe to a term deposit based on demographic, behavioral, and economic features.  
The dataset is highly imbalanced, requiring special techniques to ensure fair model performance.

## 🗂 Project Structure
- `Capstone2_Bank_Marketing.ipynb` — Main notebook (EDA, preprocessing, modeling, evaluation)
- `Capstone2_Bank_Marketing.html` — HTML export of notebook
- `Capstone2_Final_Project_Report.pdf` — Final report
- `Data/` — Dataset folder
- `README.md` — Project overview and documentation

## ⚙️ Requirements
- Python 3.10+
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- xgboost
- lightgbm
- matplotlib / seaborn

## 🔍 Dataset
- Source: UCI Machine Learning Repository – Bank Marketing Dataset  
- Target variable: `y` (term deposit subscription: yes/no)  
- Rows: 41,188  
- Features: 20+ (categorical + numerical)

## 🧠 Methodology
- Data Cleaning & Preprocessing  
- Handling categorical variables (One-Hot Encoding)  
- Scaling numerical features (StandardScaler)  
- Addressing class imbalance (SMOTE)  
- Model comparison using:
  - Logistic Regression  
  - Random Forest  
  - XGBoost  
  - LightGBM  
- Evaluation with Stratified 5-Fold Cross-Validation  

## 📊 Model Performance (Cross-Validation Summary)

| Model | Avg F1 | Avg Recall | Avg ROC AUC |
|-------|--------|------------|--------------|
| LightGBM | 0.6367 | 0.7928 | 0.9422 |
| Random Forest | 0.6247 | 0.6875 | 0.9411 |
| XGBoost | 0.6111 | 0.6479 | 0.9415 |
| Logistic Regression | 0.5821 | 0.7398 | 0.9163 |

## 🏆 Final Model (LightGBM Results)

- Precision: 0.492  
- Recall: 0.8941  
- F1-score: 0.6347  
- ROC AUC: 0.9480  

## 💡 Key Findings
- Call duration, previous campaign outcome, contact frequency, and economic indicators are the strongest predictors.
- LightGBM achieved the best balance between Recall, F1-score, and AUC.
- The model is highly effective at minimizing false negatives (important for marketing).

## 📌 Recommendations
- Prioritize customers with high predicted probability.
- Focus on improving call quality rather than increasing call volume.
- Re-target customers with positive past responses.

## 🚀 Future Work
- Hyperparameter tuning  
- Cost-sensitive learning  
- More feature engineering  
- Additional sampling techniques  
- Deploy the model as a REST API  
- Run A/B testing with real marketing campaigns  

## 👤 Author
Kevin Chen  
Springboard Data Science Career Track  
2025
