import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import math

# ---------------------------------------------
# STEP 1: Load & Preprocess Dataset
# ---------------------------------------------
df = pd.read_csv("Mall_Customers.csv")

le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])  # Male=1, Female=0
df["HighSpender"] = (df["Spending Score (1-100)"] > 50).astype(int)

X = df.drop(["CustomerID", "HighSpender"], axis=1)
y = df["HighSpender"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------
# STEP 2: Train Models
# ---------------------------------------------
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

xgb_model = xgb.XGBClassifier(eval_metric="logloss", random_state=42)
xgb_model.fit(X_train, y_train)

# ---------------------------------------------
# STEP 3: Streamlit UI
# ---------------------------------------------
st.title("💳 Customer Spending Behavior Prediction")
st.markdown("### Predict whether a customer is a **High Spender** or **Low Spender** using multiple ML algorithms.")

# User input form
st.sidebar.header("Enter Customer Details")
gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=30)
income = st.sidebar.number_input("Annual Income (k$)", min_value=0, max_value=200, value=50)
spend_score = st.sidebar.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)

# Encode Gender & Scale input
gender_encoded = 1 if gender == "Male" else 0
input_data = np.array([[gender_encoded, age, income, spend_score]])
input_scaled = scaler.transform(input_data)

# ---------------------------------------------
# STEP 4: Prediction + Formula Explanation
# ---------------------------------------------
if st.sidebar.button("Predict"):
    st.subheader("🔮 Model Predictions")

    # ---- Logistic Regression ----
    prob_lr = log_reg.predict_proba(input_scaled)[0][1]
    pred_lr = 1 if prob_lr >= 0.5 else 0

    st.markdown("#### 🧮 Logistic Regression")
    st.write(f"**Formula:** P(HighSpender) = 1 / (1 + e^(-z))")
    st.write(f"where z = w0 + w1*Gender + w2*Age + w3*Income + w4*Score")
    coefs = log_reg.coef_[0]
    intercept = log_reg.intercept_[0]
    z_value = (
        intercept +
        coefs[0]*gender_encoded +
        coefs[1]*age +
        coefs[2]*income +
        coefs[3]*spend_score
    )
    st.write(f"Computed z = {z_value:.4f}")
    st.write(f"Sigmoid( z ) = {prob_lr:.4f}")
    st.success(f"Prediction: {'High Spender 💰' if pred_lr==1 else 'Low Spender 💼'}")

    # ---- KNN ----
    pred_knn = knn.predict(input_scaled)[0]
    st.markdown("---")
    st.markdown("#### 🤝 K-Nearest Neighbors (KNN)")
    st.write("**Logic:** Finds 5 nearest customers (using Euclidean distance) and uses majority vote.")
    st.write("Distance formula:")
    st.latex(r"d = \sqrt{(x_1 - x_{i1})^2 + (x_2 - x_{i2})^2 + ...}")
    st.success(f"Prediction: {'High Spender 💰' if pred_knn==1 else 'Low Spender 💼'}")

    # ---- XGBoost ----
    prob_xgb = xgb_model.predict_proba(input_scaled)[0][1]
    pred_xgb = 1 if prob_xgb >= 0.5 else 0

    st.markdown("---")
    st.markdown("#### 🚀 XGBoost (Decision Tree Ensemble)")
    st.write("**Logic:** Combines multiple decision trees to calculate a final score, then applies sigmoid.")
    st.write("Formula:")
    st.latex(r"\hat{y} = \sigma(\sum_{k=1}^{K} f_k(x))")
    st.write(f"Predicted Probability = {prob_xgb:.4f}")
    st.success(f"Prediction: {'High Spender 💰' if pred_xgb==1 else 'Low Spender 💼'}")

    # ---- Summary Table ----
    st.markdown("---")
    results = pd.DataFrame({
        "Model": ["Logistic Regression", "KNN", "XGBoost"],
        "Predicted Class": [
            "High Spender 💰" if pred_lr==1 else "Low Spender 💼",
            "High Spender 💰" if pred_knn==1 else "Low Spender 💼",
            "High Spender 💰" if pred_xgb==1 else "Low Spender 💼",
        ],
        "Probability (if applicable)": [f"{prob_lr:.2f}", "-", f"{prob_xgb:.2f}"]
    })
    st.table(results)
