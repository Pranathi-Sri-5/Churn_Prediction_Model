import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
)
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42
MODEL_PATH = "churn_model.pkl"

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────

st.set_page_config(page_title="Telecom Churn Predictor", page_icon="📡", layout="wide")

# ─────────────────────────────────────────────────────────
# FEATURE ENGINEERING (shared by train & predict)
# ─────────────────────────────────────────────────────────


def engineer_features(df, median_charge):
    df = df.copy()

    svc_cols = [
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    for col in svc_cols:
        if col in df.columns:
            df[col] = df[col].replace(
                {"No phone service": "No", "No internet service": "No"}
            )

    df["IsFirstYear"] = (df["tenure"] <= 12).astype(int)

    df["AvgMonthlyCharge"] = df.apply(
        lambda x: (
            x["TotalCharges"] / x["tenure"] if x["tenure"] > 0 else x["MonthlyCharges"]
        ),
        axis=1,
    )

    df["FiberOpticUser"] = (df["InternetService"] == "Fiber optic").astype(int)
    df["IsMonthToMonth"] = (df["Contract"] == "Month-to-month").astype(int)

    df["HighCostLowTenure"] = (
        (df["MonthlyCharges"] > median_charge) & (df["tenure"] < 12)
    ).astype(int)

    return df


# ─────────────────────────────────────────────────────────
# MODEL TRAINING & CACHING
# ─────────────────────────────────────────────────────────


@st.cache_resource
def train_model():
    """
    Train on the IBM Telco dataset if available (telco_churn.csv),
    otherwise fall back to synthetic data with a clear warning.
    Save/load the model artifact to avoid retraining every session.
    """
    # --- Try loading saved model first ---
    if os.path.exists(MODEL_PATH):
        bundle = joblib.load(MODEL_PATH)
        return (
            bundle["model"],
            bundle["scaler"],
            bundle["feature_columns"],
            bundle["median_charge"],
            bundle["metrics"],
        )

    # --- Load real dataset if available ---
    real_data_path = "telco_churn.csv"
    using_real_data = os.path.exists(real_data_path)

    if using_real_data:
        df = pd.read_csv(real_data_path)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df.dropna(subset=["TotalCharges"], inplace=True)
        df["Churn"] = (df["Churn"] == "Yes").astype(int)

        # Drop customerID if present
        if "customerID" in df.columns:
            df.drop("customerID", axis=1, inplace=True)

        # Standardize gender encoding
        df["gender"] = (df["gender"] == "Female").astype(int)
        df["Partner"] = (df["Partner"] == "Yes").astype(int)
        df["Dependents"] = (df["Dependents"] == "Yes").astype(int)
        df["PhoneService"] = (df["PhoneService"] == "Yes").astype(int)
        df["PaperlessBilling"] = (df["PaperlessBilling"] == "Yes").astype(int)

        st.session_state["using_real_data"] = True
    else:
        st.session_state["using_real_data"] = False
        # Synthetic fallback
        np.random.seed(RANDOM_STATE)
        n = 7043

        gender = np.random.choice([0, 1], n)
        senior = np.random.choice([0, 1], n, p=[0.84, 0.16])
        partner = np.random.choice([0, 1], n)
        dependents = np.random.choice([0, 1], n)
        tenure = np.random.randint(0, 72, n)
        phone_svc = np.random.choice([0, 1], n)
        internet = np.random.choice(["DSL", "Fiber optic", "No"], n)
        online_sec = np.random.choice(["No", "Yes"], n)
        online_bkp = np.random.choice(["No", "Yes"], n)
        dev_prot = np.random.choice(["No", "Yes"], n)
        tech_sup = np.random.choice(["No", "Yes"], n)
        stream_tv = np.random.choice(["No", "Yes"], n)
        stream_mov = np.random.choice(["No", "Yes"], n)
        contract = np.random.choice(["Month-to-month", "One year", "Two year"], n)
        paperless = np.random.choice([0, 1], n)
        payment = np.random.choice(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            n,
        )
        monthly_chg = np.random.uniform(18, 119, n)
        total_chg = monthly_chg * tenure

        churn_prob = np.clip(
            0.05
            + 0.25 * (contract == "Month-to-month")
            + 0.10 * (internet == "Fiber optic")
            + 0.08 * (payment == "Electronic check")
            + 0.08 * senior
            - 0.10 * (tenure > 36),
            0,
            0.9,
        )
        churn = (np.random.rand(n) < churn_prob).astype(int)

        df = pd.DataFrame(
            {
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_svc,
                "InternetService": internet,
                "OnlineSecurity": online_sec,
                "MultipleLines": "No",
                "OnlineBackup": online_bkp,
                "DeviceProtection": dev_prot,
                "TechSupport": tech_sup,
                "StreamingTV": stream_tv,
                "StreamingMovies": stream_mov,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly_chg,
                "TotalCharges": total_chg,
                "Churn": churn,
            }
        )

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    train_median_charge = X_train["MonthlyCharges"].median()

    X_train = engineer_features(X_train, train_median_charge)
    X_test = engineer_features(X_test, train_median_charge)

    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    feature_columns = X_train.columns.tolist()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train_scaled, y_train)

    model = LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000)
    model.fit(X_res, y_res)

    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "cm": confusion_matrix(y_test, y_pred),
        "fpr": fpr,
        "tpr": tpr,
        "report": classification_report(y_test, y_pred, output_dict=True),
    }

    # Save model for future sessions
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "feature_columns": feature_columns,
            "median_charge": train_median_charge,
            "metrics": metrics,
        },
        MODEL_PATH,
    )

    return model, scaler, feature_columns, train_median_charge, metrics


# ─────────────────────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────


def predict_churn(data, model, scaler, feature_columns, median_charge):
    df = pd.DataFrame([data])
    df = engineer_features(df, median_charge)
    df = pd.get_dummies(df)

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]

    X_scaled = scaler.transform(df)
    prob = model.predict_proba(X_scaled)[0][1]
    pred = int(prob >= 0.5)
    return pred, prob


# ─────────────────────────────────────────────────────────
# RETENTION RECOMMENDATIONS
# ─────────────────────────────────────────────────────────


def get_recommendations(data, prob):
    recs = []
    if data["Contract"] == "Month-to-month":
        recs.append(
            "📋 **Offer a 1-year contract discount** — month-to-month customers churn most."
        )
    if data["InternetService"] == "Fiber optic":
        recs.append(
            "🌐 **Review Fiber pricing** — Fiber optic users show elevated churn risk."
        )
    if data["PaymentMethod"] == "Electronic check":
        recs.append(
            "💳 **Encourage auto-pay setup** — electronic check users have higher churn."
        )
    if data["SeniorCitizen"] == 1:
        recs.append(
            "👴 **Assign a dedicated support agent** — senior customers benefit from personalized care."
        )
    if data["tenure"] < 12:
        recs.append(
            "🎯 **Enroll in a loyalty rewards program** — first-year customers are high-risk."
        )
    if data["TechSupport"] == "No":
        recs.append(
            "🛠️ **Offer a free TechSupport trial** — customers without support churn more."
        )
    if data["OnlineSecurity"] == "No":
        recs.append(
            "🔒 **Bundle OnlineSecurity** — security add-ons improve retention."
        )
    if not recs:
        recs.append("✅ This customer appears stable. Maintain regular engagement.")
    return recs


# ─────────────────────────────────────────────────────────
# BATCH PREDICTION
# ─────────────────────────────────────────────────────────


def run_batch_prediction(uploaded_df, model, scaler, feature_columns, median_charge):
    results = []
    for _, row in uploaded_df.iterrows():
        try:
            data = row.to_dict()
            # Normalize types
            for col in [
                "gender",
                "SeniorCitizen",
                "Partner",
                "Dependents",
                "PhoneService",
                "PaperlessBilling",
            ]:
                if col in data:
                    val = data[col]
                    if isinstance(val, str):
                        data[col] = 1 if val.lower() in ["yes", "female", "1"] else 0
                    else:
                        data[col] = int(val)
            _, prob = predict_churn(data, model, scaler, feature_columns, median_charge)
            results.append(
                {
                    **{
                        k: v
                        for k, v in data.items()
                        if k in ["tenure", "Contract", "MonthlyCharges"]
                    },
                    "Churn Probability": f"{prob:.2%}",
                    "Risk Level": (
                        "🔴 High"
                        if prob >= 0.5
                        else ("🟡 Medium" if prob >= 0.3 else "🟢 Low")
                    ),
                }
            )
        except Exception:
            results.append({"Error": "Could not process this row"})
    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────────────────


def page_predict(model, scaler, feature_columns, median_charge):
    st.header("🔍 Predict Customer Churn")

    if st.session_state.get("using_real_data") is False:
        st.warning(
            "⚠️ **Running on synthetic data.** For real predictions, place `telco_churn.csv` "
            "(IBM Telco dataset from Kaggle) in the app directory and restart.",
            icon="⚠️",
        )

    st.markdown("Fill in all customer details below and click **Predict**.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with col2:
        st.subheader("📶 Services")
        phone_svc = st.selectbox("Phone Service", ["Yes", "No"])
        multi_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_sec = st.selectbox("Online Security", ["No", "Yes"])
        online_bkp = st.selectbox("Online Backup", ["No", "Yes"])
        dev_prot = st.selectbox("Device Protection", ["No", "Yes"])
        tech_sup = st.selectbox("Tech Support", ["No", "Yes"])
        stream_tv = st.selectbox("Streaming TV", ["No", "Yes"])
        stream_mov = st.selectbox("Streaming Movies", ["No", "Yes"])

    with col3:
        st.subheader("💳 Account Info")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=0.5)

        # Auto-calculate total charges with note
        total = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        max_value=10000.0,
        value=float(monthly * tenure),
        step=0.5
    )

    st.divider()

    if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
        data = {
            "gender": 1 if gender == "Female" else 0,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": 1 if partner == "Yes" else 0,
            "Dependents": 1 if dependents == "Yes" else 0,
            "tenure": tenure,
            "PhoneService": 1 if phone_svc == "Yes" else 0,
            "MultipleLines": multi_lines,
            "InternetService": internet,
            "OnlineSecurity": online_sec,
            "OnlineBackup": online_bkp,
            "DeviceProtection": dev_prot,
            "TechSupport": tech_sup,
            "StreamingTV": stream_tv,
            "StreamingMovies": stream_mov,
            "Contract": contract,
            "PaperlessBilling": 1 if paperless == "Yes" else 0,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }

        pred, prob = predict_churn(data, model, scaler, feature_columns, median_charge)

        st.subheader("📊 Prediction Result")

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            if pred == 1:
                st.error(f"⚠️ **HIGH CHURN RISK**\n\nProbability: **{prob:.2%}**")
            elif prob >= 0.3:
                st.warning(f"🟡 **MEDIUM CHURN RISK**\n\nProbability: **{prob:.2%}**")
            else:
                st.success(f"✅ **LOW CHURN RISK**\n\nProbability: **{prob:.2%}**")

            st.progress(prob, text=f"Risk: {prob:.2%}")

        with res_col2:
            st.subheader("💡 Retention Recommendations")
            for rec in get_recommendations(data, prob):
                st.markdown(rec)


def page_batch(model, scaler, feature_columns, median_charge):
    st.header("📂 Batch Churn Prediction")
    st.markdown(
        "Upload a CSV file with customer data to predict churn for multiple customers at once. "
        "The file should have the same columns as the Telco dataset."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(f"Loaded **{len(df)} rows** and **{len(df.columns)} columns**.")
        st.dataframe(df.head(), use_container_width=True)

        if st.button("🔮 Run Batch Prediction", type="primary"):
            with st.spinner("Predicting..."):
                results = run_batch_prediction(
                    df, model, scaler, feature_columns, median_charge
                )

            st.success(f"✅ Done! Predictions for {len(results)} customers.")
            st.dataframe(results, use_container_width=True)

            csv = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Results CSV",
                data=csv,
                file_name="churn_predictions.csv",
                mime="text/csv",
            )
    else:
        st.info("👆 Upload a CSV file to get started.")

        # Show expected columns
        with st.expander("📋 Expected CSV Columns"):
            expected = [
                "gender",
                "SeniorCitizen",
                "Partner",
                "Dependents",
                "tenure",
                "PhoneService",
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "Contract",
                "PaperlessBilling",
                "PaymentMethod",
                "MonthlyCharges",
                "TotalCharges",
            ]
            st.code(", ".join(expected))


def page_performance(metrics):
    st.header("📈 Model Performance")

    if st.session_state.get("using_real_data") is False:
        st.warning(
            "⚠️ Metrics below are based on **synthetic data** and do not reflect real-world performance.",
            icon="⚠️",
        )

    # Metric cards
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    m2.metric("Precision", f"{metrics['precision']:.2%}")
    m3.metric("Recall", f"{metrics['recall']:.2%}")
    m4.metric("F1 Score", f"{metrics['f1']:.2%}")
    m5.metric("ROC-AUC", f"{metrics['roc_auc']:.2%}")

    st.divider()

    plot_col1, plot_col2 = st.columns(2)

    with plot_col1:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            metrics["cm"],
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
        plt.close()

    with plot_col2:
        st.subheader("ROC Curve")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.plot(
            metrics["fpr"],
            metrics["tpr"],
            color="steelblue",
            lw=2,
            label=f"AUC = {metrics['roc_auc']:.3f}",
        )
        ax2.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title("ROC Curve")
        ax2.legend(loc="lower right")
        st.pyplot(fig2)
        plt.close()

    # Classification report table
    with st.expander("📋 Full Classification Report"):
        report = metrics.get("report", {})
        if report:
            report_df = pd.DataFrame(report).transpose().round(3)
            st.dataframe(report_df, use_container_width=True)


def page_about():
    st.header("ℹ️ About This App")

    st.markdown(
        """
    ## 📡 Telecom Customer Churn Predictor

    This app uses a **Logistic Regression** model (with L1 regularization and SMOTE oversampling)
    to predict the likelihood of a telecom customer churning.

    ---

    ### 🧠 Model Details

    | Property | Value |
    |---|---|
    | Algorithm | Logistic Regression (L1 / liblinear) |
    | Imbalance Handling | SMOTE oversampling |
    | Train/Test Split | 80% / 20% (stratified) |
    | Preprocessing | StandardScaler + feature engineering |

    ---

    ### 🔧 Engineered Features

    Beyond raw inputs, the model uses:
    - **IsFirstYear** — whether tenure ≤ 12 months
    - **AvgMonthlyCharge** — TotalCharges / tenure
    - **FiberOpticUser** — binary flag for fiber internet
    - **IsMonthToMonth** — binary flag for contract type
    - **HighCostLowTenure** — high monthly charge AND tenure < 12

    ---

    ### 📦 How to Use With Real Data

    1. Download the **IBM Telco Customer Churn** dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
    2. Save it as `telco_churn.csv` in the same directory as `app.py`
    3. Delete `churn_model.pkl` if it exists (to force retraining)
    4. Restart the app — it will train on real data automatically

    ---

    ### 📁 Project Structure

    ```
    telecom-churn-predictor/
    ├── app.py               # Main Streamlit app
    ├── requirements.txt     # Python dependencies
    ├── telco_churn.csv      # Real dataset (add manually)
    ├── churn_model.pkl      # Saved model (auto-generated)
    └── README.md            # Project documentation
    ```
    """
    )


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────


def main():
    st.title("📡 Telecom Customer Churn Predictor")
    st.caption("Powered by Logistic Regression with SMOTE | Built with Streamlit")

    with st.spinner("Loading model..."):
        model, scaler, feature_columns, median_charge, metrics = train_model()

    st.sidebar.title("🔀 Navigation")

    page = st.sidebar.radio(
        "Select Page",
        ["🔍 Predict Churn", "📂 Batch Prediction", "📈 Model Performance", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    st.sidebar.markdown("**Model Info**")
    st.sidebar.caption(f"Accuracy: {metrics['accuracy']:.2%}")
    st.sidebar.caption(f"ROC-AUC: {metrics['roc_auc']:.2%}")

    data_label = (
        "✅ Real data"
        if st.session_state.get("using_real_data")
        else "⚠️ Synthetic data"
    )
    st.sidebar.caption(f"Data: {data_label}")

    if page == "🔍 Predict Churn":
        page_predict(model, scaler, feature_columns, median_charge)
    elif page == "📂 Batch Prediction":
        page_batch(model, scaler, feature_columns, median_charge)
    elif page == "📈 Model Performance":
        page_performance(metrics)
    elif page == "ℹ️ About":
        page_about()


if __name__ == "__main__":
    main()
