import streamlit as st
import pandas as pd
# Removed 'shap' import
import numpy as np
import matplotlib.pyplot as plt # Added matplotlib for plotting feature importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# ----------------------------------------------------
# 1. PAGE CONFIGURATION
# ----------------------------------------------------
# Set up the basic configuration for the Streamlit page
st.set_page_config(
    page_title="Telco Churn Explorer",
    page_icon="🔎", # Using a non-emoji icon for a professional look
    layout="centered"
)

st.title("Telco Churn Explorer")
st.write("Interactively explore churn predictions using Machine Learning and feature importance.")

# ----------------------------------------------------
# 2. LOAD AND PREPARE DATA
# ----------------------------------------------------
# Use Streamlit's cache decorator to prevent reloading the data on every rerun
@st.cache_data
def load_data():
    """Loads, cleans, and prepares the Telco Churn dataset."""
    # Updated to a new, verified URL from a stable IBM repository
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    
    # Convert TotalCharges to numeric, coercing errors (empty strings are NaN)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    
    # Fill missing TotalCharges (occurs when tenure is 0) with the median
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Convert the target variable 'Churn' to numerical (1 for Yes, 0 for No)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df

df = load_data()

# Define features based on their data type
categorical_cols = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

# Define feature matrix X and target vector y
X = df[categorical_cols + numeric_cols]
y = df["Churn"]

# ----------------------------------------------------
# 3. MODEL TRAINING (Pipeline with One-Hot Encoding)
# ----------------------------------------------------
# Use Streamlit's cache_resource to store the trained model object
@st.cache_resource
def train_model(X_data, y_data, numeric_features, categorical_features):
    """Defines and trains the Random Forest model pipeline."""
    
    # 1. Preprocessor setup
    preprocessor = ColumnTransformer(
        transformers=[
            # Scale numeric features using StandardScaler
            ("num", StandardScaler(), numeric_features),
            # One-Hot Encode categorical features. sparse_output=False for compatibility.
            ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop' # Drop any features not explicitly listed above
    )

    # 2. Pipeline definition
    model_pipeline = Pipeline(steps=[
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced", # Use balanced weights to handle class imbalance
            n_jobs=-1
        ))
    ])

    # 3. Model training
    model_pipeline.fit(X_data, y_data)
    
    # Return the trained pipeline
    return model_pipeline, preprocessor

model, preprocessor = train_model(X, y, numeric_cols, categorical_cols)

# ----------------------------------------------------
# 4. SIDEBAR — USER INPUTS
# ----------------------------------------------------
st.sidebar.header("Adjust Customer Parameters")
st.sidebar.write("Configure the profile of the customer you want to analyze.")

# Numerical Inputs
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12, help="Number of months the customer has stayed with the company.")
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.25, 118.75, 70.0, help="The amount charged to the customer monthly.")
# Estimate TotalCharges based on tenure and monthly charges for simplicity
total_charges = tenure * monthly_charges

# Categorical Inputs
# Core relationship features
partner = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])

# Contract and Billing
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# Internet and Security Services
internet = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])


# Build a dictionary for prediction (using Male/No for gender/PhoneService/MultipleLines/Streaming defaults not explicitly controlled)
input_dict = {
    "gender": "Male",
    "Partner": partner,
    "Dependents": dependents,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": internet,
    "OnlineSecurity": security,
    "OnlineBackup": online_backup,
    "DeviceProtection": "No internet service" if internet == "No" else "No", # Defaulting these based on InternetService
    "TechSupport": tech_support,
    "StreamingTV": "No internet service" if internet == "No" else "No",
    "StreamingMovies": "No internet service" if internet == "No" else "No",
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}

input_df = pd.DataFrame([input_dict])

# ----------------------------------------------------
# 5. PREDICTION
# ----------------------------------------------------
# Get the probability of churn (class 1)
prediction_prob = model.predict_proba(input_df)[0][1]
# Get the hard prediction (0 or 1)
prediction = model.predict(input_df)[0]

st.subheader("Predicted Churn Probability")

# Display the probability with a metric widget
st.metric(
    label="Churn Likelihood",
    value=f"{prediction_prob * 100:.2f}%",
    delta=None
)

# Provide a clear interpretation of the prediction
if prediction == 1:
    st.error("This customer is likely to churn. Recommended action: Proactive retention strategy.")
else:
    st.success("This customer is unlikely to churn. Recommended action: Standard account monitoring.")

# ----------------------------------------------------
# 6. FEATURE IMPORTANCE VISUALIZATION (SHAP Alternative)
# ----------------------------------------------------
st.subheader("Model Feature Importance")
st.write("This chart shows the global importance of each feature in the Random Forest model.")

# Get feature names after OHE
ohe_feature_names = model["pre"].named_transformers_["cat"].get_feature_names_out(categorical_cols)
all_feature_names = list(numeric_cols) + list(ohe_feature_names)

# Extract feature importances from the trained Random Forest classifier
importances = model["clf"].feature_importances_

# Create a Series of feature importances
feature_importances = pd.Series(importances, index=all_feature_names)

# Select top 15 features for clarity and sort them
top_features = feature_importances.sort_values(ascending=False).head(15)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))
top_features.plot.barh(ax=ax, color='teal')
ax.set_title("Top 15 Model Feature Importances (Gini Index)")
ax.set_xlabel("Importance Score")
ax.set_ylabel("Feature")
plt.gca().invert_yaxis() # Display the most important feature at the top
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(fig)