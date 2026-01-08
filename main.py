import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc)
from sklearn.neighbors import NearestCentroid
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score

data = pd.read_csv("C:/Users/jajal/Downloads/ML project.csv")
data.info()
data = data.sample(n=3000, random_state=42).copy()
data = data.copy()

# Separate features (X) and target variable (y)
#ID is just a unique tag per loan, irrelavent

X = data.drop(['Default', 'LoanID'], axis=1)
y = data['Default']

# Because we are trying to find the most significant correlations with another categorical variable ('Default'), it is very important to ensure we encode our categorical to ensure accurate feature selection.
# One-hot encode all object (categorical) columns
X_encoded = pd.get_dummies(X, columns=X.select_dtypes(include=['object']).columns, drop_first=True)


rf_regressor = RandomForestRegressor(n_estimators=10, random_state=42)
rf_regressor.fit(X_encoded, y)
feature_importances = rf_r
egressor.feature_importances_


importance_df = pd.DataFrame({'Feature': X_encoded.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)


plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='maroon')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
plt.gca().invert_yaxis()
plt.show()

X = data[['InterestRate', 'Income', 'LoanAmount', 'Age', 'CreditScore',
          'MonthsEmployed', 'DTIRatio', 'LoanTerm', 'NumCreditLines']]
y = data['Default']
y.value_counts(normalize=True)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
y_resampled.value_counts(normalize=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

classifiers = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=10, random_state=42),
    'SVM': SVC(kernel='linear', random_state=42, probability=True),  # Enable probability estimates
    'k-NN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'XGBoost': xgb.XGBClassifier(n_estimators=10, random_state=42),

}

# Initialize dictionaries to store evaluation metric results
results = {
    'Classifier': [],
    'Accuracy': [],
    'F1 Score': [],
    'Precision': [],
    'Recall': [],
    'ROC AUC': []
}

# Loop through classifiers and calculate various evaluation metrics using cross-validation
for classifier_name, classifier in classifiers.items():
    y_scores = cross_val_predict(classifier, X_scaled, y_resampled, cv=5, method='predict_proba')[:, 1]

    accuracy_scores = cross_val_score(classifier, X_scaled, y_resampled, cv=5, scoring='accuracy')
    f1_scores = cross_val_score(classifier, X_scaled, y_resampled, cv=5, scoring='f1')
    precision_scores = cross_val_score(classifier, X_scaled, y_resampled, cv=5, scoring='precision')
    recall_scores = cross_val_score(classifier, X_scaled, y_resampled, cv=5, scoring='recall')
    roc_auc_scores = cross_val_score(classifier, X_scaled, y_resampled, cv=5, scoring='roc_auc')

    # Take the mean of cross-validation scores
    accuracy_mean = np.mean(accuracy_scores)
    f1_mean = np.mean(f1_scores)
    precision_mean = np.mean(precision_scores)
    recall_mean = np.mean(recall_scores)
    roc_auc_mean = np.mean(roc_auc_scores)

    results['Classifier'].append(classifier_name)
    results['Accuracy'].append(accuracy_mean)
    results['F1 Score'].append(f1_mean)
    results['Precision'].append(precision_mean)
    results['Recall'].append(recall_mean)
    results['ROC AUC'].append(roc_auc_mean)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_resampled, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{classifier_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{classifier_name} ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

# Create line plots to compare evaluation metrics
plt.figure(figsize=(12, 6))
for metric_name, metric_results in {
    'Accuracy': results['Accuracy'],
    'F1 Score': results['F1 Score'],
    'Precision': results['Precision'],
    'Recall': results['Recall'],
    'ROC AUC': results['ROC AUC']
}.items():

    plt.plot(results['Classifier'], metric_results, label=metric_name, marker='o')

plt.xlabel('Classifiers')
plt.ylabel('Score')
plt.title('Classifier Comparison (Cross-Validation)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

results_df = pd.DataFrame(results)
print(results_df)
# Define a function to select the best model based on the evaluation results
def select_best_model(results_df):
    # Sort the results DataFrame by the desired metric (e.g., ROC AUC, Accuracy, etc.)
    best_model = results_df.sort_values(by='ROC AUC', ascending=False).iloc[0]
    return best_model

# Select the best model
best_model = select_best_model(results_df)
print("Best Model:")
print(best_model)

# Instantiate the selected model
selected_classifier = classifiers[best_model['Classifier']]

# Train the selected model on the entire resampled dataset
selected_classifier.fit(X_scaled, y_resampled)

# Function to interactively predict labels for user-provided samples
def predict_label(sample, model):
    # Sample should be a dictionary containing feature values
    sample_df = pd.DataFrame(sample, index=[0])
    # Preprocess the sample (encode categorical features, scale numerical features)
    #sample_encoded = pd.get_dummies(sample_df, columns=X.columns, drop_first=True)  # Use all X columns
    sample_scaled = scaler.transform(sample_df)
    # Predict the label
    prediction = model.predict(sample_scaled)[0]
    return prediction

# Interactive prediction loop
S_COUNT=int(input("ENTER HOW MANY SAMPLES YOU WANT TO PREDICT : "))
for i in range(S_COUNT):
    print(f"Enter The details/contents of the sample-{i+1} :")
    sample = {}
    for feature in X.columns:
        value = input(f"Enter value for {feature}: ")
        if value.isdigit():  # Convert numeric inputs to float
            value = float(value)
        sample[feature] = value
    if 'exit' in sample.values():
        break
    # Predict label using the selected model
    prediction = predict_label(sample, selected_classifier)
    print("Predicted Label:", prediction)
    print("")
    print("For the given sample the loan default is : ",prediction==1)
    print("")
    