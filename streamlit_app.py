import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ----------------------------------------------------
# 1. PAGE CONFIGURATION
# ----------------------------------------------------
st.set_page_config(
    page_title="Telco Churn Explorer",
    page_icon="📉",
    layout="centered"
)

st.title("📉 Telco Churn Explorer")
st.write("Interactively explore churn predictions using ML + SHAP!")

# ----------------------------------------------------
# 2. LOAD DATA
# ----------------------------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/blastchar/telco-customer-churn/master/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    
    # Clean TotalCharges (some have spaces)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Convert target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df

df = load_data()

# Features
categorical_cols = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

X = df[categorical_cols + numeric_cols]
y = df["Churn"]

# ----------------------------------------------------
# 3. MODEL TRAINING (pipeline)
# ----------------------------------------------------
@st.cache_resource
def train_model():
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", "passthrough", categorical_cols)  # RandomForest can handle encoded passthrough
        ]
    )

    model = Pipeline(steps=[
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    model.fit(X, y)
    return model, preprocessor

model, preprocessor = train_model()

# ----------------------------------------------------
# 4. SIDEBAR — USER INPUTS
# ----------------------------------------------------
st.sidebar.header("Adjust Customer Parameters")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges", 10, 120, 60)
total_charges = tenure * monthly_charges

internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment = st.sidebar.selectbox("Payment Method", ["Electronic check", "Credit card", "Bank transfer", "Mailed check"])
security = st.sidebar.selectbox("Online Security", ["Yes", "No"])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])

# Build a dictionary for prediction
input_dict = {
    "gender": "Male",
    "Partner": partner,
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": internet,
    "OnlineSecurity": security,
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": contract,
    "PaperlessBilling": "Yes",
    "PaymentMethod": payment,
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}

input_df = pd.DataFrame([input_dict])

# ----------------------------------------------------
# 5. PREDICTION
# ----------------------------------------------------
prediction_prob = model.predict_proba(input_df)[0][1]
prediction = model.predict(input_df)[0]

st.subheader("🔮 Predicted Churn Probability")
st.metric(
    label="Churn Likelihood",
    value=f"{prediction_prob * 100:.2f}%",
    delta=None
)

if prediction == 1:
    st.error("⚠️ This customer is likely to churn.")
else:
    st.success("✅ This customer is unlikely to churn.")

# ----------------------------------------------------
# 6. SHAP EXPLAINER
# ----------------------------------------------------
st.subheader("🧠 SHAP Feature Explanation")

explainer = shap.TreeExplainer(model["clf"])
shap_values = explainer.shap_values(model["pre"].transform(input_df))

# Plot SHAP force plot
shap.initjs()
st.write("### Contribution of each feature to the prediction:")

force_plot_html = shap.force_plot(
    explainer.expected_value[1],
    shap_values[1],
    input_df,
    matplotlib=False
)

# Render SHAP in Streamlit
st.components.v1.html(force_plot_html.html(), height=300, scrolling=True)
