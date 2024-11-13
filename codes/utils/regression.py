import time

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, median_absolute_error, r2_score

def define_models(model_names, params):
    methods = {}
    for model_name in model_names:
        model_key = "_".join(model_name.lower().split(" "))

        # Linear Model
        if(model_name=="Linear Regression"):
            methods[model_name] = LinearRegression()
        elif(model_name=="Ridge Regression"):
            methods[model_name] = Ridge(random_state=42)
        elif(model_name=="Lasso"):
            methods[model_name] = Lasso(random_state=42)
        elif(model_name=="Elastic Net"):
            methods[model_name] = ElasticNet(random_state=42)
        
        # SVM
        elif(model_name=="SVR"):
            methods[model_name] = SVR()
        elif(model_name=="Linear SVR"):
            methods[model_name] = LinearSVR(random_state=42)
        
        # Neighbors
        elif(model_name=="KNN"):
            methods[model_name] = KNeighborsRegressor()

        # Tree
        elif(model_name=="Decision Tree"):
            max_depth = params[f"{model_key}_params_max_depth"]
            methods[model_name] = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        elif(model_name=="Extra Trees"):
            max_depth = params[f"{model_key}_params_max_depth"]
            methods[model_name] = ExtraTreesRegressor(max_depth=max_depth, random_state=42)
        elif(model_name=="Random Forest"):
            max_depth = params[f"{model_key}_params_max_depth"]            
            methods[model_name] = RandomForestRegressor(max_depth=max_depth, random_state=42)
        elif(model_name=="AdaBoost"):
            methods[model_name] = AdaBoostRegressor(random_state=42)
        elif(model_name=="Gradient Boosting"):
            max_depth = params[f"{model_key}_params_max_depth"]            
            methods[model_name] = GradientBoostingRegressor(max_depth=max_depth, random_state=42)
        
    return methods

def train_function(model, X_train, y_train):
    start = time.time()
        
    model.fit(X_train, y_train)

    end = time.time()
    train_runtime = round(end-start, 2)

    return model, train_runtime

def predict_function(model, X):
    start = time.time()

    y_pred = model.predict(X)

    end = time.time()
    pred_runtime = round(end-start, 2)

    return y_pred, pred_runtime

def adjusted_r2_score(y_test, y_pred, n, p):
    score = r2_score(y_test, y_pred)
    return 1 - (1 - score) * (n - 1) / (n - p - 1)

def get_results(y_true, y_pred, metric_df, metrics, split, n, p):
    for metric in metrics:
        if(metric=="MSE"):
            score = mean_squared_error(y_true, y_pred) # MSE
        elif(metric=="RMSE"):
            score = root_mean_squared_error(y_true, y_pred) # RMSE
        elif(metric=="MAE"):
            score = mean_absolute_error(y_true, y_pred) # MAE
        elif(metric=="MedAE"):
            score = median_absolute_error(y_true, y_pred) # MedAE
        elif(metric=="R2"):
            score = r2_score(y_true, y_pred) # R-squared
        elif(metric=="Adj R2"):
            score = adjusted_r2_score(y_true, y_pred, n, p) # Adjusted R-squared
        metric_df.loc[0, f"{split} {metric}"] = score

    return metric_df