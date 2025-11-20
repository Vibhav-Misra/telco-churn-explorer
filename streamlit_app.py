import streamlit as st
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Telco Churn Explorer",
    page_icon="🔎",
    layout="centered"
)

st.title("Telco Churn Explorer")
st.write("Interactively explore churn predictions using Machine Learning and SHAP feature explanations.")

@st.cache_data
def load_data():
    """Loads, cleans, and prepares the Telco Churn dataset."""
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df

df = load_data()

categorical_cols = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

X = df[categorical_cols + numeric_cols]
y = df["Churn"]

@st.cache_resource
def train_model(X_data, y_data, numeric_features, categorical_features):
    """Defines and trains the Random Forest model pipeline."""
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop' 
    )

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
    
    # Return both the trained pipeline and the preprocessor (needed for SHAP)
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
# 6. SHAP EXPLAINER
# ----------------------------------------------------
st.subheader("SHAP Feature Explanation")
st.write("The following visualization shows how each feature contributes to the final churn prediction, pushing it toward Churn (red) or No Churn (blue).")

# Initialize the SHAP TreeExplainer using the Random Forest Classifier part of the pipeline
explainer = shap.TreeExplainer(model["clf"])

# Get the feature names from the OneHotEncoder step
ohe_feature_names = model["pre"].named_transformers_["cat"].get_feature_names_out(categorical_cols)
all_feature_names = list(numeric_cols) + list(ohe_feature_names)

# Transform the input data using the trained preprocessor
input_transformed = model["pre"].transform(input_df)

# Calculate SHAP values for the transformed input
# shap_values[1] is for the positive class (Churn=1)
shap_values = explainer.shap_values(input_transformed)

# Create a DataFrame for the SHAP force plot input
# Note: Input must be the transformed data, but the feature names map to the original OHE columns
shap_input_df = pd.DataFrame(input_transformed, columns=all_feature_names)

# Plot SHAP force plot for the Churn (1) class
shap.initjs()

force_plot_html = shap.force_plot(
    explainer.expected_value[1],
    shap_values[1],
    shap_input_df,
    matplotlib=False
)

# Render SHAP in Streamlit using st.components.v1.html
st.components.v1.html(force_plot_html.html(), height=350, scrolling=True)