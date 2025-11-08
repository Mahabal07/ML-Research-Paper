# 💳 Customer Spending Behavior Prediction (Streamlit App)

This project predicts whether a mall customer is a **High Spender** or **Low Spender** using **Machine Learning algorithms** — Logistic Regression, KNN, and XGBoost — and explains **how each model makes its prediction**.

---

## 🚀 Features

- Upload or use your local **Mall_Customers.csv** dataset  
- Predict spending behavior based on:
  - Gender  
  - Age  
  - Annual Income (k$)  
  - Spending Score (1–100)  
- See predictions from **three algorithms** side-by-side:
  - 🧮 Logistic Regression  
  - 🤝 K-Nearest Neighbors  
  - 🚀 XGBoost  
- Displays:
  - Formula-based explanation for each algorithm  
  - Probability scores  
  - Final predicted class (High/Low Spender)  

---

## 🧠 Algorithms Used

| Algorithm | Logic | Formula / Decision |
|------------|--------|--------------------|
| **Logistic Regression** | Linear + Sigmoid | `P = 1 / (1 + e^(-z))` where `z = w0 + w1*Gender + w2*Age + w3*Income + w4*Score` |
| **K-Nearest Neighbors (KNN)** | Distance-based voting | Majority vote among k-nearest customers using Euclidean distance |
| **XGBoost** | Tree-based ensemble | Combines multiple decision trees → final probability = `sigmoid(sum of trees)` |

---

## 📦 Project Structure

📁 ML-Spending-Predictor
│
├── streamlit_app.py # Main Streamlit app
├── Mall_Customers.csv # Dataset file (place in same directory)
├── requirements.txt # Python dependencies
└── README.md # Project documentation

🧾 Example Prediction Logic
Logistic Regression:
z = w0 + w1*Gender + w2*Age + w3*Income + w4*SpendingScore
P = 1 / (1 + e^-z)
If P >= 0.5 → High Spender
Else → Low Spender

KNN:
Calculate Euclidean distance from all known customers
Pick K nearest ones (default 5)
If majority are High Spenders → Predict High Spender
Else → Low Spender

XGBoost:
Each decision tree gives a small prediction
Final Score = Sum of all tree outputs
P = Sigmoid(Final Score)

📊 Output Example

When you click "Predict", you’ll see:

The computed z value and sigmoid probability for Logistic Regression

The KNN logic explanation

XGBoost probability output

A summary table comparing all model results

Example:

Model	Predicted Class	Probability
Logistic Regression	High Spender 💰	0.87
KNN	High Spender 💰	-
XGBoost	High Spender 💰	0.90


🧠 Future Enhancements
  > Deploy on Streamlit Cloud or Hugging Face Spaces

  > Add feature importance visualization

  > Allow users to train models with their own data

  > Predict Spending Score directly (regression model)
