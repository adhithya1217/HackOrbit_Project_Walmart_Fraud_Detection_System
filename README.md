# HackOrbit_Project_Walmart_Fraud_Detection_System

## Live On Streamlit : https://hackorbitprojectwalmartfrauddetectionsystem-7rvrusmffw6yazbpwx.streamlit.app

A machine learning-powered fraud detection system built to identify potentially fraudulent transactions in retail environments, inspired by real-world e-commerce scenarios such as those faced by Walmart.
---

##  Project Overview
In this project, we analyze transactional data to:
- Detect fraudulent activities
- Handle data imbalance using SMOTE
- Train a Random Forest classifier
- Evaluate the model with precision, recall, and F1-score
---

Project Deployed Succesfully on Streamlit.app

## Features implemented in this project are :
✅ 1. Real-Time Fraud Prediction
Users can input transaction data manually.

The model predicts whether the transaction is fraudulent or legitimate.

Uses a trained Random Forest model with a customizable threshold slider.

✅ 2. Data Upload & Dynamic Support
Users can upload their own CSV files with similar schema.

App adapts dynamically to uploaded data for predictions and visualization.

✅ 3. Interactive Visualizations (Plotly)
📊 Fraud Count by Hour – Bar chart to show hourly fraud trends.

🧭 Fraud by Customer Location – Heatmap style bar chart by region.

🛒 Fraud by Product Category – Pie chart.

💰 Transaction Amount Histogram – Compare fraud vs non-fraud.

✅ 4. Feature Importance (Model Insights)
Displays top 15 important features from the model.

Helps interpret model behavior using Plotly horizontal bar charts.

✅ 5. Live Probability Gauge
A real-time meter gauge shows the fraud probability (%) using Plotly.

Intuitive display for understanding how risky a transaction is.

✅ 6. Light/Dark Mode Toggle
UI supports both Light and Dark themes.

Automatically defaults to Dark mode with toggle in sidebar.

✅ 7. Clean & Modern Streamlit UI
Uses custom styling with CSS inside st.markdown.

Responsive layout with 3-column inputs, buttons, and hover animations.

## 🔍 Why This Project Stands Out:

1.Real-Time + Business-Focused: Predicts fraud instantly with inputs tailored to retail (Walmart-like) cases.

2.Customizable Risk: Threshold slider lets users tune fraud sensitivity as per business needs.

3.Polished UI: Dark mode by default + theme toggle for better UX.

4.Interactive Analytics: Plotly charts for fraud by hour, location, category & animated gauge meter.

5.Explainable AI: Shows feature importance—no black-box predictions.

6.Upload-Ready: Accepts user CSVs and adapts automatically.

7.Scalable Codebase: Modular backend, supports model upgrades easily.

## This project doesn’t just predict fraud — it lets users adjust the risk threshold, instantly see results visually (with a gauge + Plotly charts), and understand why a prediction was made through feature importance insights — all inside a professional, theme-adaptive interface.

## commit 1 :
Set up project structure

Imported libraries (pandas, Streamlit, Plotly, etc.)

Loaded and cleaned initial CSV dataset

Basic EDA and saved a cleaned (.csv) file.

## commit 2 :
Built and trained the Random Forest model

Performed class balancing

Evaluated with accuracy, precision, recall

Serialized model using joblib as balanced_rf_model.joblib

## commit 3:
Created Streamlit app layout

Added input form for manual predictions

Loaded model and predicted fraud from form inputs

Displayed prediction with simple status messages

## commit 4:
Integrated sidebar toggle to switch between Light and Dark themes

Dynamically styled buttons, backgrounds, and text

## commit 5:
Added real-time interactive charts using Plotly:

Fraud by hour

Fraud by product category

Fraud by customer location

Amount distribution (fraud vs. non-fraud)

## commit 6:
Integrated live gauge chart to visualize fraud probability

Used plotly.graph_objects.Indicator

## commit 7:
Extracted feature importances from trained Random Forest model

Visualized top 15 features contributing to predictions

Enabled users to interpret the model's logic

## commit 8:
Enabled CSV file upload from sidebar

Added threshold slider for real-time prediction control

Cleaned UI layout, added captions, tooltips, icons

Ensured dynamic visuals adapt to uploaded datasets

