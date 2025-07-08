# HackOrbit_Project_Walmart_Fraud_Detection_System



A machine learning-powered fraud detection system built to identify potentially fraudulent transactions in retail environments, inspired by real-world e-commerce scenarios such as those faced by Walmart.
---

##  Project Overview
## 1st Commit
In this project, we analyze transactional data to:
- Detect fraudulent activities
- Handle data imbalance using SMOTE
- Train a Random Forest classifier
- Evaluate the model with precision, recall, and F1-score
---

##  Features

- **Data Preprocessing**: Cleaned and structured synthetic transactional data
- **SMOTE Oversampling**: Addressed class imbalance in fraud detection
- **Random Forest Model**: Trained to classify transactions as fraudulent or not
- **Model Evaluation**: Classification report & confusion matrix

##2nd Commit
1. Class Imbalance Analysis
We began by examining the class distribution of the IsFraud column.
2. Balancing Data with SMOTE
To address the imbalance, we used SMOTE (Synthetic Minority Oversampling Technique), which generates synthetic samples of the minority class (IsFraud = 1):
 3. Feature Encoding
We handled categorical variables using One-Hot Encoding, converting text labels into numerical columns using pd.get_dummies():
4. Train-Test Split
The resampled data was split into training and test sets:
5. Random Forest Classifier
We trained a Random Forest Classifier, known for its robustness and high accuracy:
6. Saving the Trained Model
The trained model was saved using joblib for future use or deployment.
