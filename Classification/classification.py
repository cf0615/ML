# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from xgboost import XGBClassifier

# Load preprocessed data
preprocessed_data = pd.read_csv('preprocess_dataset.csv')

# Separate features and target
X = preprocessed_data.drop('Status', axis=1)
y = preprocessed_data['Status']

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------- Logistic Regression Tuning ----------------------
# Define parameter grid for Logistic Regression
param_grid_log_reg = {
    'penalty': ['l1', 'l2'],  # Regularization type
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'solver': ['liblinear', 'saga']  # Solver options (liblinear supports l1 and l2)
}

# Initialize RandomizedSearchCV for Logistic Regression
random_search_log_reg = RandomizedSearchCV(
    LogisticRegression(max_iter=5000),  # Increase iterations
    param_distributions=param_grid_log_reg,
    n_iter=10,
    scoring='roc_auc',
    cv=5,
    random_state=42
)

# Fit to the training data
random_search_log_reg.fit(X_train, y_train)

# Best parameters and performance for Logistic Regression
print("Best Parameters for Logistic Regression:", random_search_log_reg.best_params_)
print("Best ROC AUC Score:", random_search_log_reg.best_score_)

# Use the best model
best_log_reg = random_search_log_reg.best_estimator_

# ---------------------- Random Forest Tuning ----------------------
# Define parameter grid for Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 300, 400, 500],  # Number of trees
    'max_depth': [None, 10, 20, 30, 40],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples to split a node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples at a leaf node
    'max_features': ['sqrt', 'log2']  # Maximum features considered for splitting
}

# Initialize RandomizedSearchCV for Random Forest
random_search_rf = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_distributions=param_grid_rf,
                                      n_iter=10, scoring='roc_auc', cv=5, random_state=42, n_jobs=-1)

# Fit to the training data
random_search_rf.fit(X_train, y_train)

# Best parameters and performance for Random Forest
print("Best Parameters for Random Forest:", random_search_rf.best_params_)
print("Best ROC AUC Score:", random_search_rf.best_score_)

# Use the best model
best_rf = random_search_rf.best_estimator_

# ---------------------- Evaluation After Tuning ----------------------
# Logistic Regression Evaluation
y_pred_log_reg_tuned = best_log_reg.predict(X_test)
print("\nTuned Logistic Regression Evaluation:")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_log_reg_tuned))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_log_reg_tuned))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred_log_reg_tuned))
print("\nROC AUC Score:", roc_auc_score(y_test, y_pred_log_reg_tuned))

# Plot ROC Curve for Logistic Regression
fpr_log_reg, tpr_log_reg, _ = roc_curve(y_test, best_log_reg.predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(fpr_log_reg, tpr_log_reg, color='b', label=f'Logistic Regression (AUC = {roc_auc_score(y_test, y_pred_log_reg_tuned):.2f})')
plt.plot([0, 1], [0, 1], color='r', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Logistic Regression')
plt.legend()
plt.show()

# Random Forest Evaluation
y_pred_rf_tuned = best_rf.predict(X_test)
print("\nTuned Random Forest Evaluation:")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf_tuned))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf_tuned))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred_rf_tuned))
print("\nROC AUC Score:", roc_auc_score(y_test, y_pred_rf_tuned))

# Plot ROC Curve for Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, best_rf.predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(fpr_rf, tpr_rf, color='b', label=f'Random Forest (AUC = {roc_auc_score(y_test, y_pred_rf_tuned):.2f})')
plt.plot([0, 1], [0, 1], color='r', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Random Forest')
plt.legend()
plt.show()

