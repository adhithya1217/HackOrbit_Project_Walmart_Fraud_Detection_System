import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- Layout & Theme Setup ---
# --- Layout & Theme Setup ---
st.set_page_config(page_title="Walmart Fraud Detection", layout="wide")

theme_mode = st.sidebar.radio("🌗 Choose Theme", ["Dark", "Light"], index=0)

primary_color = "#ff9800" if theme_mode == "Dark" else "#1a73e8"
background_color = "#1e1e1e" if theme_mode == "Dark" else "#f7f9fc"
text_color = "#f5f5f5" if theme_mode == "Dark" else "#333333"


st.markdown(f"""
    <style>
        .main {{ background-color: {background_color}; }}
        .stButton>button {{
            background-color: {primary_color};
            color: white;
            padding: 10px 24px;
            border-radius: 8px;
            transition: 0.3s ease;
        }}
        .stButton>button:hover {{
            background-color: #0b5ed7;
        }}
        h1, h2, h3, h4, h5, h6, p {{
            color: {text_color};
        }}
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("🛡️ Walmart Fraud Detection System")
st.caption("Detect fraudulent transactions in real time using AI.")

# --- Load Data & Model ---
@st.cache_resource
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Data load error: {e}")
        return None

@st.cache_resource
def load_model(file_path):
    try:
        return joblib.load(file_path)
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

df = load_data("synthetic_realistic_fraud_200.csv")
model = load_model("balanced_rf_model.joblib")

if df is None or model is None:
    st.stop()

# --- Upload Section ---
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV", type=["csv"])
if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        st.sidebar.success("Data uploaded successfully!")
    else:
        st.stop()

# --- Threshold Slider ---
threshold = st.sidebar.slider("🎯 Prediction Threshold", 0.0, 1.0, 0.5, 0.01)

# --- Data Overview ---
st.header("📊 Data Overview")
with st.expander("📋 View Sample & Summary"):
    st.subheader("🔹 Sample Transactions")
    st.dataframe(df.head())
    st.subheader("🔸 Statistical Summary")
    st.write(df.describe())

# --- Interactive Graphs ---
st.header("📈 Interactive Visualizations")
with st.expander("📊 Explore Fraud Patterns"):
    st.subheader("🕓 Fraud Count by Hour of Day")
    fraud_by_hour = df[df["IsFraud"] == 1]["HourOfDay"].value_counts().sort_index()
    fig1 = px.bar(x=fraud_by_hour.index, y=fraud_by_hour.values,
                  labels={"x": "Hour", "y": "Fraud Count"},
                  title="Fraud by Hour", color_discrete_sequence=["indianred"])
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("📦 Fraud by Product Category")
    fraud_by_cat = (
        df[df["IsFraud"] == 1]["ProductCategory"]
        .value_counts()
        .reset_index()
    )
    fraud_by_cat.columns = ["ProductCategory", "Count"]
    fig2 = px.pie(fraud_by_cat, names="ProductCategory", values="Count",
                  title="Fraud by Product Category", hole=0.4)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📍 Fraud by Customer Location")
    fraud_geo = df[df["IsFraud"] == 1]["CustomerLocation"].value_counts().reset_index()
    fraud_geo.columns = ["Location", "Count"]
    fig3 = px.bar(fraud_geo, x="Location", y="Count", color="Count", title="Fraud by Location",
                  color_continuous_scale="reds")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("💰 Transaction Amount Distribution")
    fig4 = px.histogram(df, x="Amount", color="IsFraud", nbins=40,
                        title="Amount Histogram: Fraud vs Non-Fraud")
    st.plotly_chart(fig4, use_container_width=True)

# --- Input Form ---
st.header("🤖 Predict Transaction Fraud")
st.write("Enter transaction details:")

cols = st.columns(3)
with cols[0]:
    amount = st.number_input("Amount", 0.0, value=float(df["Amount"].mean()))
    return_count = st.number_input("Return Count", 0, value=int(df["ReturnCount"].mean()))
    coupon_used = st.selectbox("Coupon Used", [0, 1])
    hour_of_day = st.number_input("Hour of Day", 0, 23, value=int(df["HourOfDay"].mean()))
    day_of_week = st.number_input("Day of Week", 0, 6, value=int(df["DayOfWeek"].mean()))

with cols[1]:
    month = st.number_input("Month", 1, 12, value=int(df["Month"].mean()))
    tsl = st.number_input("Time Since Last Transaction", 0.0, value=float(df["TimeSinceLastTransaction"].mean()))
    tx_last_hour = st.number_input("Transactions in Last Hour", 0, value=int(df["TransactionsInLastHour"].mean()))
    amt_last_hour = st.number_input("Amount in Last Hour", 0.0, value=float(df["AmountInLastHour"].mean()))
    age = st.number_input("Customer Age", 0, value=int(df["CustomerAge"].mean()))

with cols[2]:
    price = st.number_input("Product Price", 0.0, value=float(df["ProductPrice"].mean()))
    device_id = st.selectbox("Device ID", df["DeviceID"].unique())
    location = st.selectbox("Customer Location", df["CustomerLocation"].unique())
    category = st.selectbox("Product Category", df["ProductCategory"].unique())

# --- Prediction ---
if st.button("🔍 Predict Fraud"):
    input_df = pd.DataFrame({
        "Amount": [amount],
        "DeviceID": [device_id],
        "ReturnCount": [return_count],
        "CouponUsed": [coupon_used],
        "HourOfDay": [hour_of_day],
        "DayOfWeek": [day_of_week],
        "Month": [month],
        "TimeSinceLastTransaction": [tsl],
        "TransactionsInLastHour": [tx_last_hour],
        "AmountInLastHour": [amt_last_hour],
        "CustomerAge": [age],
        "ProductPrice": [price],
        "CustomerLocation": [location],
        "ProductCategory": [category]
    })

    cat_cols = ["DeviceID", "CustomerLocation", "ProductCategory"]
    input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)
    df_encoded_cols = pd.get_dummies(df.drop("IsFraud", axis=1), columns=cat_cols, drop_first=True).columns
    input_final = input_encoded.reindex(columns=df_encoded_cols, fill_value=0)

    pred_prob = model.predict_proba(input_final)[:, 1]
    is_fraud = int(pred_prob[0] >= threshold)

    st.subheader("📌 Prediction Result")
    if is_fraud:
        st.error(f"🚨 Fraud Detected! Block this account urgently... (Probability: {pred_prob[0]:.4f})")
    else:
        st.success(f"✅ Legitimate Transaction (Probability: {pred_prob[0]:.4f})")

    # --- Real-Time Gauge Chart ---
    st.subheader("📟 Real-Time Fraud Probability Gauge")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(pred_prob[0]*100, 2),
        title={"text": "Fraud Probability (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "crimson"},
            "steps": [
                {"range": [0, 50], "color": "lightgreen"},
                {"range": [50, 80], "color": "orange"},
                {"range": [80, 100], "color": "red"},
            ]
        }
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # --- Feature Importance ---
    st.subheader("🧠 Feature Importance (Model Insight)")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_names = df_encoded_cols
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(15)

        fig_feat = px.bar(importance_df, x="Importance", y="Feature",
                          orientation="h", title="Top 15 Important Features",
                          color="Importance", color_continuous_scale="viridis")
        st.plotly_chart(fig_feat, use_container_width=True)
    else:
        st.info("⚠️ Feature importance not supported by this model.")
