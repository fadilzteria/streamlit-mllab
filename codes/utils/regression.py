import time

from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import streamlit as st

@st.cache_data
def define_models(model_names, n_models, params):
    methods = {}
    for model_name in model_names:
        model_key = "_".join(model_name.lower().split(" "))
        nums = n_models[f"{model_key}_n_models"]
        for n in range(1, nums+1):
            model_n_name = f"{model_name} {n}"
            model_n_key = "_".join(model_n_name.lower().split(" "))

            # Linear Model
            if model_name=="Linear Regression":
                methods[model_n_name] = LinearRegression()
            elif model_name=="Ridge Regression":
                methods[model_n_name] = Ridge(random_state=42)
            elif model_name=="Lasso":
                methods[model_n_name] = Lasso(random_state=42)
            elif model_name=="Elastic Net":
                methods[model_n_name] = ElasticNet(random_state=42)

            # SVM
            elif model_name=="SVR":
                methods[model_n_name] = SVR()
            elif model_name=="Linear SVR":
                methods[model_n_name] = LinearSVR(random_state=42)

            # Neighbors
            elif model_name=="KNN":
                methods[model_n_name] = KNeighborsRegressor()

            # Tree
            elif model_name=="Decision Tree":
                max_depth = params[f"{model_n_key}_params_max_depth"]
                methods[model_n_name] = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
            elif model_name=="Extra Trees":
                max_depth = params[f"{model_n_key}_params_max_depth"]
                methods[model_n_name] = ExtraTreesRegressor(max_depth=max_depth, random_state=42)
            elif model_name=="Random Forest":
                max_depth = params[f"{model_n_key}_params_max_depth"]
                methods[model_n_name] = RandomForestRegressor(max_depth=max_depth, random_state=42)
            elif model_name=="AdaBoost":
                methods[model_n_name] = AdaBoostRegressor(random_state=42)
            elif model_name=="Gradient Boosting":
                max_depth = params[f"{model_n_key}_params_max_depth"]
                methods[model_n_name] = GradientBoostingRegressor(
                    max_depth=max_depth, random_state=42
                )

            # Advanced Ensemble Tree
            elif model_name=="XGBoost":
                xgb_fix_params = {
                    "device": "cpu", "verbosity": 0, "tree_method": "hist",
                    "random_state": 42, "enable_categorical": True, "objective": "reg:squarederror"
                }
                methods[model_n_name] = XGBRegressor(**xgb_fix_params)
            elif model_name=="LightGBM":
                lgb_fix_params = {
                    "device": "cpu", "verbose": -1, "random_state": 42, "objective": "regression"
                }
                methods[model_n_name] = LGBMRegressor(**lgb_fix_params)
            elif model_name=="CatBoost":
                cab_fix_params = {
                    "task_type": "CPU", "devices": 0, "verbose": 0,
                    "random_state": 42, "objective": "RMSE"
                }
                methods[model_n_name] = CatBoostRegressor(**cab_fix_params)

    return methods

@st.cache_data
def regress_metrics(metric, y_true, y_pred, n, p):
    if metric=="MSE":
        score = mean_squared_error(y_true, y_pred) # MSE
    elif metric=="RMSE":
        score = root_mean_squared_error(y_true, y_pred) # RMSE
    elif metric=="MAE":
        score = mean_absolute_error(y_true, y_pred) # MAE
    elif metric=="MedAE":
        score = median_absolute_error(y_true, y_pred) # MedAE
    elif metric=="R2":
        score = r2_score(y_true, y_pred) # R-squared
    else:
        score = adjusted_r2_score(y_true, y_pred, n, p) # Adjusted R-squared

    return score

def train_function(model, model_name, x_train, y_train):
    start = time.time()

    if "CatBoost" in model_name:
        cat_cols = list(x_train.select_dtypes(include=['category']).columns.values)
        # cat_cols = x_train.columns.tolist()

        x_train[cat_cols] = x_train[cat_cols].astype('str', copy=False)

        updated_params = {"cat_features": cat_cols}
        model.set_params(**updated_params)

    model.fit(x_train, y_train)

    end = time.time()
    train_runtime = round(end-start, 2)

    return model, train_runtime

def predict_function(model, model_name, data):
    start = time.time()

    if "CatBoost" in model_name:
        cat_cols = list(data.select_dtypes(include=['category']).columns.values)
        # cat_cols = data.columns.tolist()

        for cat_feat in cat_cols:
            if data[cat_feat].isnull().values.any():
                data[cat_feat] = data[cat_feat].cat.add_categories("Missing").fillna("Missing")

        data[cat_cols] = data[cat_cols].astype('str', copy=False)

    y_pred = model.predict(data)

    end = time.time()
    pred_runtime = round(end-start, 2)

    return y_pred, pred_runtime

@st.cache_data
def adjusted_r2_score(y_test, y_pred, n, p):
    score = r2_score(y_test, y_pred)
    return 1 - (1 - score) * (n - 1) / (n - p - 1)

@st.cache_data
def get_results(y_true, y_pred, metric_df, metrics, split, n, p):
    for metric in metrics:
        score = regress_metrics(metric, y_true, y_pred, n, p)
        metric_df.loc[0, f"{split} {metric}"] = score

    return metric_df
