# Student Performance Predictor

## Project Overview

The Student Performance Predictor is a Machine Learning project that predicts a student's final marks based on study hours, attendance, previous scores, and assignment scores.

This project was developed as part of a Data Science internship to demonstrate data analysis, visualization, machine learning, and model evaluation.

## Objective

The objective of this project is to analyze student-related data and build a machine learning model that predicts final marks.

## Dataset

The dataset contains 30 student records with the following features:

- Study Hours
- Attendance
- Previous Score
- Assignment Score
- Final Marks

The target variable is **Final Marks**.

## Exploratory Data Analysis

The project includes analysis of:

- Study Hours vs Final Marks
- Attendance vs Final Marks
- Feature Correlations
- Actual vs Predicted Marks

## Machine Learning Model

A **Random Forest Regressor** was used to predict final marks.

### Features

- Study Hours
- Attendance
- Previous Score
- Assignment Score

### Target

- Final Marks

The dataset was divided into:

- 80% Training Data
- 20% Testing Data

## Model Evaluation

The model was evaluated using:

- MAE
- MSE
- RMSE
- R² Score

### Results

- MAE: **0.59**
- MSE: **0.72**
- RMSE: **0.85**
- R² Score: **0.99**

## Example Prediction

For a student with:

- Study Hours: 5
- Attendance: 90%
- Previous Score: 78
- Assignment Score: 85

The model predicted:

**Final Marks: 82.67**

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Project Structure

```text
Student-Performance-Predictor/
│
├── data/
│   └── student_performance.csv
│
├── screenshots/
│   ├── study_hours_vs_marks.png
│   ├── attendance_vs_marks.png
│   ├── correlation_heatmap.png
│   ├── actual_vs_predicted.png
│   └── model_performance.png
│
├── ├── student_performance.py
├── requirements.txt
└── README.md
```

## Author

**Aleeza Qaiser**

Data Science Internship Project