# Loan-Defaulter-Prediction-using-ML-Algorithms

📊**Loan Defaulter Prediction using Machine Learning Algorithms**

This project focuses on predicting loan default risk using supervised machine learning techniques. The objective is to build and evaluate multiple classification models to identify borrowers who are likely to default, helping financial institutions improve risk assessment and decision-making.

The pipeline includes data preprocessing, feature importance analysis, class imbalance handling, model comparison using cross-validation, and ROC-based evaluation.

**🧠 Problem Statement**

Loan default prediction is a critical task in the financial domain. Given borrower demographic and financial attributes, the goal is to classify whether a loan applicant is likely to default (1) or not (0).

Key challenges addressed:

Handling imbalanced datasets

Identifying important predictive features

Comparing multiple ML algorithms fairly

🗂️ **Dataset**

The dataset consists of borrower-related financial and demographic features.

Target variable: Default (binary classification)

Irrelevant identifiers (e.g., LoanID) are removed.

Categorical features are one-hot encoded.

A random sample of 3000 records is used for experimentation.

⚙️ **Methodology**
1️⃣ **Data Preprocessing**

Removal of irrelevant features (LoanID)

One-hot encoding of categorical variables

Feature scaling using StandardScaler

Handling class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)

2️⃣ **Feature Importance Analysis**

A Random Forest Regressor is used to estimate feature importance, helping identify the most influential factors contributing to loan default prediction.

3️⃣ **Machine Learning Models Used**

The following classifiers are trained and evaluated:

Decision Tree

Random Forest

Support Vector Machine (Linear Kernel)

k-Nearest Neighbors

Naive Bayes

XGBoost

4️⃣ **Model Evaluation**

Models are evaluated using 5-fold cross-validation with the following metrics:

Accuracy

Precision

Recall

F1 Score

ROC-AUC

5️⃣ **Best Model Selection**

The model with the highest ROC-AUC score is automatically selected and trained on the full resampled dataset.

🔮 Prediction Module

An interactive prediction module allows users to input feature values manually and receive a real-time loan default prediction based on the selected best model.

🛠️ **Technologies & Libraries**

Programming Language: Python

Libraries:

NumPy, Pandas

Scikit-learn

XGBoost

Imbalanced-learn (SMOTE)

Matplotlib

📈 **Results**

The project demonstrates that ensemble-based models (Random Forest, XGBoost) generally outperform simpler classifiers.

SMOTE significantly improves recall and ROC-AUC for minority class detection.

The selected best model achieves strong predictive performance across multiple evaluation metrics.


👩‍💻 Author

Thrisha Reddy Jajala
B.Tech Computer Science (Data Science)
GitHub: https://github.com/thrisha1217
