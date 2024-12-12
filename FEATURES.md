# More Features

## Feature Engineering

- Feature creation
  - Create math features
    - Arithmetic operations between two features (addition, subtraction, absolute subtraction, multiplication, true division, floor division)
    - Squared and cubed
    - Multipled by 10 and 100
  - Create categorical group features
- Feature extraction
  - Extract datetime columns (year, month, hour, etc.)
- Drop categorical columns with many unique counts
- Categorical encoding
  - One-hot encoding
  - Categorical for XGBoost, LightGBM, and CatBoost
- Feature scaling with StandardScaler()

## Machine Learning Algorithms

| ML Algorithm | ML Tasks | Availabled Parameters |
| ------------ | -------- | --------------------- |
| Logistic Regression | Classification | - |
| Linear Discriminant Analysis | Classification | - |
| Linear Regression | Regression | - |
| Ridge Regression | Regression | - |
| Lasso | Regression | - |
| Elastic Net | Regression | - |
| Bernoulli Bayes | Classification | - |
| Gaussian Bayes | Classification | - |
| KNN | Classification and Regression | n_neighbors |
| SVM | Classification and Regression | - |
| Linear SVM | Classification and Regression | - |
| Decision Tree | Classification and Regression | max_depth |
| Extra Trees | Classification and Regression | max_depth |
| Random Forest | Classification and Regression | max_depth |
| AdaBoost | Classification and Regression | - |
| Gradient Boosting | Classification and Regression | max_depth |
| XGBoost | Classification and Regression | - |
| LightGBM | Classification and Regression | - |
| CatBoost | Classification and Regression | - |

## Metrics

| Classification | Regression |
| -------------- | ---------- |
| Accuracy | MSE (Mean Squared Error) |
| Precision | RMSE (Root Mean Squared Error) |
| Recall | MAE (Mean Absolute Error) |
| F1 Score | MedAE (Median Absolute Error) |
| ROC AUC | R<sup>2</sup> |
| Average Precision | Adjusted R<sup>2</sup> |