# 📉 Customer Churn Prediction App

An interactive **Machine Learning + Streamlit web app** that predicts whether a telecom customer will **churn (leave)** or **stay** using a complete end-to-end pipeline built with **Logistic Regression**.

🔗 **[Live App Demo](https://churnpredictionmodel-6acabzhc7fsmuva88w66up.streamlit.app/)** : For Real-Time Engagement  


---

## Project Overview

Customer churn is a critical problem in the telecom industry. This project builds a robust ML pipeline to:
- Analyze customer data  
- Handle class imbalance  
- Train an optimized model  
- Predict churn probability  
- Provide business insights  

The solution is deployed as an **interactive Streamlit application** where users can upload datasets and get real-time predictions.

---

## Machine Learning Pipeline

```text
Load → Clean → Split → Engineer → Encode → Scale → 
Handle Imbalance (SMOTE) → Feature Selection (L1) → 
Cross Validation → Hyperparameter Tuning → 
Threshold Optimization → Evaluation
```
--- 

##  Features
- Data Handling
- Data Preprocessing
- Feature Engineering
- Encoding & Scaling
- Class Imbalance Handling
- Model Optimization
- Threshold Adjustment
- Model Evaluation
- Visualizations
---
## Results (on Telco Dataset)

| Metric       | Score |
|--------------|------:|
| Accuracy     | 0.737 |
| Recall       | 0.810 |
| Precision    | 0.503 |
| F1 Score     | 0.620 |
| ROC-AUC      | 0.838 |

### Insights
- High **recall (0.810)** → model captures most churn customers  
- Moderate **precision (0.503)** → some false positives exist  
- Balanced **F1 Score (0.620)** indicates overall stable performance  
- Strong **ROC-AUC (0.838)** shows good class separation ability  

---
## Run Locally
1. Clone the repo
git clone https://github.com/YOUR_USERNAME/customer-churn-predictor.git
cd customer-churn-predictor

2. Install dependencies
pip install -r requirements.txt

3. Run the app
streamlit run app.py

---
## Data Format

The application expects a **CSV file** with customer-related features.

### Columns include

```text
customerID, gender, SeniorCitizen, Partner, Dependents,
tenure, PhoneService, MultipleLines, InternetService,
OnlineSecurity, OnlineBackup, DeviceProtection,
TechSupport, StreamingTV, StreamingMovies,
Contract, PaperlessBilling, PaymentMethod,
MonthlyCharges, TotalCharges, Churn
```
---
## Tech Stack

- **Programming Language:** Python  
- **Data Processing:** Pandas, NumPy  
- **Machine Learning:** Scikit-learn  
- **Imbalance Handling:** Imbalanced-learn (SMOTE)  
- **Visualization:** Matplotlib, Seaborn  
- **Web App Framework:** Streamlit

## License
MIT
