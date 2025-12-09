# Predicting High Income Using Machine Learning

This project uses the Adult Census Income dataset (1994 U.S. Census, downloaded from Kaggle) to predict whether an individual earns **> $50K annually** using machine learning models.

---

## 1. Project Objectives

- Build a binary classifier to predict high vs. low income.  
- Identify key socioeconomic factors influencing income.  
- Compare Logistic Regression with XGBoost and explain model behavior using SHAP.

---

## 2. Data & Preprocessing

- Replaced missing values (`'?'`) with `NaN`.  
- Grouped certain categorical variables and applied one-hot encoding.  
- Performed an **80/20 stratified split** and standardized features where needed.

---

## 3. Models & Key Results

### **Logistic Regression (Baseline)**
- Accuracy ~0.85; weaker at detecting high-income individuals.
- Limited by dataset imbalance and non-linear relationships.

### **XGBoost**
- Accuracy ~0.89; better precision/recall for high-income class.
- Captures complex interactions Logistic Regression cannot.

---

## 4. Model Interpretation (SHAP)

Top influential features:
- **Marital status (married)**  
- **Education level**  
- **Capital gain**  
- **Age**  
- **Hours worked per week**

SHAP visualizations highlight how these factors push predictions toward high or low income.

---

## 5. Dependencies

Core libraries:
- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `scikit-learn`
- `xgboost`
- `shap`
- `kagglehub` (for dataset download)

Install all with:

```bash
pip install numpy pandas scikit-learn xgboost shap kagglehub
