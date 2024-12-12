# Streamlit ML Lab: Accessible, Fast, and Explainable Machine Learning

<p align="center">
  <img src="assets\main_1.png" width="40%"/>
  <img src="assets\main_2.png" width="40%"/>
  <img src="assets\main_3.png" width="40%"/>
  <img src="assets\main_4.png" width="40%"/>
</p>

## Introduction
Streamlit ML Lab is a Python-based application built with Streamlit that enables everyone to
apply data science and machine learning practices ***without writing code***. The application features a ***simple, interactive user interface*** with ***various configurations***, supported by ***popular data science and machine learning libraries***.

Major Features:

- 👋 **Accessibility**  
  Offers opportunities for individuals with little or no technical experiences through a user-friendly interface, designed for easy installation and use.
- 🚄 **Quick Experimentation**  
  Focuses on speed and simplicity in experimenting with various configurations in data preprocessing and the machine learning process, leveraging Streamlit's interactive design.
- 📊 **Comprehensive Insights**  
  Provides detailed analyses with clear, understandable visualizations of datasets and model results, helping identify areas for improvement in experiments.
- 🔎 **Explainability**  
  Delivers global and individual modal explanations using SHAP and scikit-learn libraries, helping interpret model behavior effectively.
- 🔬 **ML Tasks and Algorithms**  
  Supports common ML Tasks such as classification and regression, which address many real-world problems, along with various popular machine learning algorithms.

## Table of Content
- [Streamlit ML Lab: Accessible, Fast, and Explainable Machine Learning](#streamlit-ml-lab-accessible-fast-and-explainable-machine-learning)
  - [Introduction](#introduction)
  - [Table of Content](#table-of-content)
  - [Get Started](#get-started)
    - [Download](#download)
    - [Installation](#installation)
    - [Run the Application](#run-the-application)
  - [Features](#features)
  - [Guides](#guides)
  - [Future Interesting Features](#future-interesting-features)
  - [References](#references)
  - [License](#license)

## Get Started

The application is designed for easy installation in the local computer so that everyone is free to use this without advanced hardware specifications. Run this application in the terminal.

> [!NOTE]
> This code is run with Windows powershell

### Download

Clone the repository

```bash
# Download the GitHub Code
git clone https://github.com/fadilzteria/streamlit-mllab

# Move to the Application Folder
cd streamlit-mllab
```

### Installation

Create the virtual environment. Make sure that [VENV] virtual environment name is defined and changed

```powershell
# Create Virtual Environment
python -m virtualenv [VENV]

# Activate Virtual Environment
[VENV]/Scripts/Activate.ps1

# Upgrade Pip
python -m pip install --upgrade pip

# Install Requirements
pip install -r requirements.txt
```

### Run the Application

After installing, run this code. Then, the Streamlit application will be displayed in the screen

```powershell
# Run Streamlit Application
python -m streamlit run codes/app.py
```

## Features

| Section                  | Description | Features |
| ------------------------ | ----------- | -------- |
| Data Preprocessing       | Check the data quality and preprocess from raw into cleaned dataset | <ul><li>Input the raw dataset</li><li>Show the data quality (unique counts for each feature, missing values, etc.)</li><li>Preprocess the dataset</li></ul> |
| EDA: Univariate Analysis | Explore each feature in the cleaned dataset | <ul><li>Categorical features (value counts)</li><li>Numerical features (statistics, box plot, and distribution)</li></ul> |
| EDA: Bivariate Analysis  | Explore the relationship between features in the cleaned dataset | <ul><li>Correlation Matrix</li><li>The relationship between numerical features (Pearson, Spearman, or Kendall's correlation)</li><li>The relationship between numerical and categorical feature (statistical inference with ANOVA and Partial Eta-squared)</li><li>The relationship between categorical features (statistical inference with Chi-squared, Phi Coefficient, and Cramer's Value)</li></ul> |
| Model Training           | Train the model with completed configuration | <ul><li>Various ML tasks (classification, multi-classification, and regression)</li><li>Stratified k-fold cross validation</li><li>Feature engineering configuration (categorical encoding, scaling, etc.)</li><li>Model variations and configuration (linear model-based, tree-based, advanced ensemble tree-based &rarr; XGBoost, LightGBM, CatBoost, etc.)</li><li>Metrics configuration</li></ul> |
| Model Evaluation         | Evaluate the trained model from metrics based on train and valid dataset | <ul><li>Metric results (overall, class-wise for multi-classifcation, categorical-based)</li><li>Confusion matrix for classification</li><li>Training runtime</li><li>Regression diagnostics for regression</li><li>Predicted distribution</li></ul> |
| Model Explainability     | Interpret and explain the trained model with visualization | <ul><li>Global explanations (feature importances and partial dependence plots)</li><li>Easy-difficult samples</li><li>Local explanations for valid dataset</li></ul> |
| Model Prediction         | Predict labels from the test dataset with trained models | <ul><li>Input the w/wo labeled test dataset</li><li>Preprocess the test dataset</li><li>Data comparison between train and test dataset</li><li>Predict and get labels with trained models</li><li>Get metrics from the labeled test dataset</li><li>Extract local explanations for test dataset</li></ul> |

See the [more features](FEATURES.md) for detailed feature engineering techniques, ML algorithms and availabled parameters, and metrics

## Guides

To understand more about how to use this application, open [guides](GUIDES.md)

## Future Interesting Features

- [ ] More model parameters
- [ ] Other feature engineering techniques (drop unnecessary features, cap outliers)
- [ ] Model training scenarios (optimization, blending)
- [ ] Other ML tasks (multi-label, time-series)
- [ ] Multivariate analysis (dimensionality reduction, clustering)

Excited to see if there are more ideas that can be integrated in the application!

## References

These references support the creation of this project as a simple and interactive application, making it possible for everyone to become a creative experimenter!

- Streamlit Documentation &rarr; https://docs.streamlit.io
- SHAP Documentation &rarr; https://shap.readthedocs.io/en/latest
- Python popular libraries for data science and machine learning &rarr; NumPy, pandas, Matplotlib, seaborn, and scikit-learn

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.