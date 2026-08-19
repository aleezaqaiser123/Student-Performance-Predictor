import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# 1. Load Dataset
# -----------------------------

data = pd.read_csv("data/student_performance.csv")

print("Student Performance Dataset")
print(data.head())


# -----------------------------
# 2. Data Information
# -----------------------------

print("\nDataset Information:")
print(data.info())

print("\nMissing Values:")
print(data.isnull().sum())


# -----------------------------
# 3. Exploratory Data Analysis
# -----------------------------

print("\nStatistical Summary:")
print(data.describe())

print("\nCorrelation with Final Marks:")
print(data.corr()["Final Marks"].sort_values(ascending=False))


# Study Hours vs Final Marks
plt.figure(figsize=(8, 5))
sns.scatterplot(data=data, x="Study Hours", y="Final Marks")
plt.title("Study Hours vs Final Marks")
plt.xlabel("Study Hours")
plt.ylabel("Final Marks")
plt.tight_layout()
plt.show()


# Attendance vs Final Marks
plt.figure(figsize=(8, 5))
sns.scatterplot(data=data, x="Attendance", y="Final Marks")
plt.title("Attendance vs Final Marks")
plt.xlabel("Attendance (%)")
plt.ylabel("Final Marks")
plt.tight_layout()
plt.show()


# Correlation Heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()


# -----------------------------
# 4. Prepare Data
# -----------------------------

X = data[[
    "Study Hours",
    "Attendance",
    "Previous Score",
    "Assignment Score"
]]

y = data["Final Marks"]


# -----------------------------
# 5. Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# -----------------------------
# 6. Train Machine Learning Model
# -----------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# -----------------------------
# 7. Make Predictions
# -----------------------------

predictions = model.predict(X_test)


# -----------------------------
# 8. Model Evaluation
# -----------------------------

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("-----------------------------")
print("MAE :", round(mae, 2))
print("MSE :", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 2))


# -----------------------------
# 9. Actual vs Predicted
# -----------------------------

results = pd.DataFrame({
    "Actual Marks": y_test.values,
    "Predicted Marks": predictions.round(2)
})

print("\nActual vs Predicted Marks:")
print(results)
# -----------------------------
# 11. Actual vs Predicted Graph
# -----------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    range(len(y_test)),
    y_test.values,
    marker="o",
    label="Actual Marks"
)

plt.plot(
    range(len(predictions)),
    predictions,
    marker="o",
    label="Predicted Marks"
)

plt.title("Actual vs Predicted Marks")
plt.xlabel("Test Student")
plt.ylabel("Marks")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# 10. Prediction Example
# -----------------------------

student = pd.DataFrame({
    "Study Hours": [5],
    "Attendance": [90],
    "Previous Score": [78],
    "Assignment Score": [85]
})

predicted_marks = model.predict(student)

print("\nExample Student Prediction:")
print("Predicted Final Marks:", round(predicted_marks[0], 2))