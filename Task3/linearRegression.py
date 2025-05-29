# -*- coding: utf-8 -*-
"""
Created on Thu May 29 13:18:07 2025

@author: capl2
"""

import pandas as pd

# Load dataset
df = pd.read_csv('Housing.csv')
print(df.head())
print(df.info())
print(df.describe())
# Convert categorical variables using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Check for missing values
print(df.isnull().sum())
from sklearn.model_selection import train_test_split

# Assume 'price' is the target
X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R² Score: {r2}")

import matplotlib.pyplot as plt
# Replace 'area' with your chosen single feature
X = df[['area']]
y = df['price']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Plot
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Line')
plt.xlabel("Area (sq ft)")
plt.ylabel("Price")
plt.title("Simple Linear Regression: Area vs Price")
plt.legend()
plt.show()

#multiple regressions
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()
import pandas as pd

coeff_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
coeff_df.plot(x='Feature', y='Coefficient', kind='bar', title='Feature Coefficients')
plt.tight_layout()
plt.show()
