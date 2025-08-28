import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)
import numpy as np

# 📥 Download and unzip dataset
os.makedirs("data", exist_ok=True)
os.system("kaggle datasets download -d dansbecker/melbourne-housing-snapshot -p data")
import zipfile

with zipfile.ZipFile("data/melbourne-housing-snapshot.zip", 'r') as zip_ref:
    zip_ref.extractall("data")


# 📄 Load dataset
df = pd.read_csv("data/melb_data.csv")

# 🔍 Quick check
print(df.head())
print(df.info())

# 🧹 Data Preprocessing
# Drop rows with missing values for simplicity
df.dropna(inplace=True)

# Features and target
X = df.drop(["Price", "Address", "SellerG", "Method", "Date", "Postcode", "CouncilArea", "Regionname"], axis=1)
y = df["Price"]

# Convert categorical variables
X = pd.get_dummies(X, drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🌲 Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 🎯 Predictions
y_pred = model.predict(X_test)

# 📊 Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 Evaluation Metrics:")
print(f"MAE (Mean Absolute Error): {mae:,.2f}")
print(f"RMSE (Root Mean Squared Error): {rmse:,.2f}")
print(f"R² Score: {r2:.4f}")

# 📈 Actual vs Predicted Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

# 📉 Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True)
plt.title("Residuals Distribution")
plt.xlabel("Prediction Error")
plt.show()

# 🔥 Feature Importance
importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_features, y=top_features.index)
plt.title("Top 10 Feature Importances")
plt.show()
