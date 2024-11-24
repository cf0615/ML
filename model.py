import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load the dataset
dataset_cleaned = pd.read_csv('preprocessed_dataset.csv')

# Select all the features after one-hot encoding
features = dataset_cleaned.drop(columns=['Price'])  # Drop 'Price' column to get all features
target = dataset_cleaned['Price']  # Target variable remains the same

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

#---------------------------------------------------#
# Linear Regression

# Initialize Linear Regression
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

# Make predictions
y_pred_linear = linear_reg.predict(X_test)

# Evaluate the Linear Regression model
mae_linear = mean_absolute_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print(f"Linear Regression - MAE: {mae_linear}, RMSE: {rmse_linear}, MSE: {mse_linear}, R-squared: {r2_linear}")

#---------------------------------------------------#
# XGBoost

# Define the parameter grid for GridSearchCV
param_grid_xgb = {
    'n_estimators': [100, 200, 300], #number of tress
    'learning_rate': [0.01, 0.1, 0.2], #how much each tree contributes to the final prediction
    'max_depth': [3, 5, 7], #tree complexity
    'subsample': [0.8, 1.0], #percentage of data used
    'colsample_bytree': [0.8, 1.0] #percentage of features used
}

# Initialize XGBRegressor
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Initialize GridSearchCV for XGBoost
grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)

# Fit the model using GridSearchCV to find the best hyperparameters
grid_search_xgb.fit(X_train, y_train)

# Get the best parameters
best_params_xgb = grid_search_xgb.best_params_
print(f"Best parameters for XGBoost: {best_params_xgb}")

# Train the XGBoost model with the best parameters
best_xgb_model = xgb.XGBRegressor(**best_params_xgb, objective='reg:squarederror', random_state=42)
best_xgb_model.fit(X_train, y_train)

# Make predictions
y_pred_xgb_tuned = best_xgb_model.predict(X_test)

# Evaluate the tuned XGBoost model
mae_xgb_tuned = mean_absolute_error(y_test, y_pred_xgb_tuned)
rmse_xgb_tuned = np.sqrt(mean_squared_error(y_test, y_pred_xgb_tuned))
mse_xgb_tuned = mean_squared_error(y_test, y_pred_xgb_tuned)
r2_xgb_tuned = r2_score(y_test, y_pred_xgb_tuned)

print(f"Tuned XGBoost - MAE: {mae_xgb_tuned}, RMSE: {rmse_xgb_tuned}, MSE: {mse_xgb_tuned}, R-squared: {r2_xgb_tuned}")

print("\nConclusion:")
# Evaluate Linear Regression
print(f"Linear Regression - MAE: {mae_linear}, MSE: {mse_linear}, RMSE: {rmse_linear}, R-squared: {r2_linear}")
# Evaluate XGBoost
print(f"Tuned XGBoost - MAE: {mae_xgb_tuned}, MSE: {mse_xgb_tuned}, RMSE: {rmse_xgb_tuned}, R-squared: {r2_xgb_tuned}")
