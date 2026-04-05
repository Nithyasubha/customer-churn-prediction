#  Customer Churn Prediction Project

##  Project Overview

This project focuses on predicting customer churn for a telecom company using machine learning models.
Churn prediction helps businesses identify customers who are likely to leave, enabling proactive retention strategies.

---

##  Problem Statement

Customer churn is a major challenge for telecom companies. The goal of this project is to:

* Analyze customer data
* Identify patterns leading to churn
* Build predictive models to classify customers as **Churn / Not Churn**

---

##  Dataset

* Dataset: Telecom Customer Churn Dataset
* Contains customer details such as:

  * Demographics (gender, senior citizen)
  * Account information (tenure, contract type)
  * Services (internet, phone, streaming)
  * Billing (monthly charges, total charges)
  * Target variable: **Churn**

---

##  Technologies Used

* Python
* Pandas & NumPy (Data Processing)
* Scikit-learn (ML Models)
* XGBoost (Advanced Model)
* Joblib (Model Saving)

---

##  Project Workflow

### 1. Data Preprocessing

* Handle missing values
* Convert categorical variables to numeric
* Feature scaling using StandardScaler

### 2. Model Building

Trained multiple models:

* Logistic Regression
* Random Forest Classifier
* XGBoost Classifier

### 3. Model Evaluation

Evaluated using:

* Accuracy
* ROC-AUC Score
* Confusion Matrix

---

##  Model Performance

| Model         | Accuracy | ROC-AUC |
| ------------- | -------- | ------- |
| Logistic      | ~0.78    | ~0.83   |
| Random Forest | ~0.78    | ~0.81   |
| XGBoost       | ~0.77    | ~0.81   |

 **Best Model:** Logistic Regression (based on ROC-AUC)

---

##  Confusion Matrix Example

```
[[915 118]
 [181 193]]
```

* True Negatives: 915
* False Positives: 118
* False Negatives: 181
* True Positives: 193

---

##  Model Saving

The best-performing model is saved using Joblib:

```
model/churn_model.pkl
```

---

##  Project Structure

```
churn_prediction_project/
│
├── data/
│   └── telecom_churn.csv
│
├── model/
│   ├── train.py
│   ├── preprocessing.py
│   └── churn_model.pkl
│
└── README.md
```

---

## ▶️ How to Run

1. Clone the repository:

```
git clone <your-repo-link>
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the training script:

```
python model/train.py
```

---
##  Key Insights

* Customers with **month-to-month contracts** are more likely to churn
* Higher **monthly charges** increase churn probability
* Customers with **long tenure** are less likely to churn

---

##  Future Improvements

* Hyperparameter tuning (GridSearchCV)
* Feature importance visualization
* Deploy using Streamlit
* Add real-time prediction API

---

##  Conclusion

This project demonstrates an end-to-end machine learning pipeline:

* Data preprocessing
* Model training
* Evaluation
* Deployment-ready model

It is suitable for showcasing skills in **data analytics, machine learning, and problem-solving**.

---

## 👤 Author

**Nithya subha**

---


