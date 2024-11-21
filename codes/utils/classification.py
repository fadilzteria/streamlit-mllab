import time
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

def define_models(model_names, n_models, params, nunique):
    methods = {}
    for model_name in model_names:
        model_key = "_".join(model_name.lower().split(" "))
        nums = n_models[f"{model_key}_n_models"]
        for n in range(1, nums+1):
            model_n_name = f"{model_name} {n}"
            model_n_key = "_".join(model_n_name.lower().split(" "))

            # Linear Model
            if(model_name=="Logistic Regression"):
                methods[model_n_name] = LogisticRegression(random_state=42)
            
            # Discriminant Analysis
            elif(model_name=="Linear Discriminant Analysis"):
                methods[model_n_name] = LinearDiscriminantAnalysis()

            # Naive Bayes
            elif(model_name=="Bernoulli Bayes"):
                methods[model_n_name] = BernoulliNB()
            elif(model_name=="Gaussian Bayes"):
                methods[model_n_name] = GaussianNB()

            # SVM
            elif(model_name=="SVC"):
                methods[model_n_name] = SVC(probability=True)
            elif(model_name=="Linear SVC"):
                methods[model_n_name] = LinearSVC(random_state=42)

            # Neighbors
            elif(model_name=="KNN"):
                methods[model_n_name] = KNeighborsClassifier()

            # Tree
            elif(model_name=="Decision Tree"):
                max_depth = params[f"{model_n_key}_params_max_depth"]
                methods[model_n_name] = DecisionTreeClassifier(max_depth=max_depth, random_state=42, class_weight='balanced')
            elif(model_name=="Extra Trees"):
                max_depth = params[f"{model_n_key}_params_max_depth"]
                methods[model_n_name] = ExtraTreesClassifier(max_depth=max_depth, random_state=42, class_weight='balanced')
            elif(model_name=="Random Forest"):
                max_depth = params[f"{model_n_key}_params_max_depth"]            
                methods[model_n_name] = RandomForestClassifier(max_depth=max_depth, random_state=42, class_weight='balanced')
            elif(model_name=="AdaBoost"):
                methods[model_n_name] = AdaBoostClassifier(random_state=42)
            elif(model_name=="Gradient Boosting"):
                max_depth = params[f"{model_n_key}_params_max_depth"]            
                methods[model_n_name] = GradientBoostingClassifier(max_depth=max_depth, random_state=42)

            # Advanced Ensemble Tree
            elif(model_name=="XGBoost"):
                xgb_fix_params = {
                    "device": "cpu", "verbosity": 0, "tree_method": "hist", 
                    "random_state": 42, "enable_categorical": True
                }
                xgb_fix_params["objective"] = "binary:logistic" if(nunique==2) else "multi:softmax"
                methods[model_n_name] = XGBClassifier(**xgb_fix_params)
            elif(model_name=="LightGBM"):
                lgb_fix_params = {"device": "cpu", "verbose": -1, "random_state": 42}
                lgb_fix_params["objective"] = "binary" if(nunique==2) else "multiclass"
                methods[model_n_name] = LGBMClassifier(**lgb_fix_params)
            elif(model_name=="CatBoost"):
                cab_fix_params = {
                    "task_type": "CPU", "devices": 0, "verbose": 0, "random_state": 42,
                    'n_estimators': 100, 
                }
                cab_fix_params["objective"] = "Logloss" if(nunique==2) else "MultiClass"
                methods[model_n_name] = CatBoostClassifier(**cab_fix_params)

    return methods

def classif_metrics(metric, y_true, y_pred, y_pred_proba, class_names=None):
    class_scores = None
    if(metric=="Accuracy"): # Accuracy
        score = accuracy_score(y_true, y_pred) 
    elif(metric=="Precision"): # Precision
        if(class_names): 
            score = precision_score(y_true, y_pred, average="macro") 
            class_scores = precision_score(y_true, y_pred, average=None)
        else:
            score = precision_score(y_true, y_pred) 
    elif(metric=="Recall"): # Recall
        if(class_names): 
            score = recall_score(y_true, y_pred, average="macro") 
            class_scores = recall_score(y_true, y_pred, average=None)
        else:
            score = recall_score(y_true, y_pred) 
    elif(metric=="F1 Score"): # F1 Score
        if(class_names): 
            score = f1_score(y_true, y_pred, average="macro") 
            class_scores = f1_score(y_true, y_pred, average=None)
        else:
            score = f1_score(y_true, y_pred) 
    elif(metric=="ROC AUC"): # ROC AUC
        if(class_names): 
            score = roc_auc_score(y_true, y_pred_proba, multi_class='ovr') 
            class_scores = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average=None)
        else:
            score = roc_auc_score(y_true, y_pred_proba[:, 1]) 
    elif(metric=="Avg Precision"): # Avg Precision
        if(class_names): 
            y_true_classes = np.unique(y_true)
            y_true_onehot = label_binarize(y_true, classes=y_true_classes)

            score = average_precision_score(y_true_onehot, y_pred_proba)
            class_scores = average_precision_score(y_true_onehot, y_pred_proba, average=None)
        else:
            score = average_precision_score(y_true, y_pred_proba[:, 1]) 

    return score, class_scores

def train_function(model, model_name, X_train, y_train):
    start = time.time()
    
    if(any(tree_name in model_name for tree_name in ["XGBoost", "LGBM", "CatBoost"])):
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
        min_sample_weight = np.min(sample_weights)
        sample_weights *= (100/min_sample_weight)
        sample_weights = sample_weights.astype(int)

        if("CatBoost" in model_name):
            cat_cols = list(X_train.select_dtypes(include=['category']).columns.values)
            # cat_cols = X_train.columns.tolist()

            X_train[cat_cols] = X_train[cat_cols].astype('str', copy=False)

            updated_params = {"cat_features": cat_cols}
            model.set_params(**updated_params)
        
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)

    end = time.time()
    train_runtime = round(end-start, 2)

    return model, train_runtime

def predict_function(model, model_name, X):
    start = time.time()

    if("CatBoost" in model_name):
        cat_cols = list(X.select_dtypes(include=['category']).columns.values)
        # cat_cols = X.columns.tolist()

        for cat_feat in cat_cols:
            if(X[cat_feat].isnull().values.any()):
                X[cat_feat] = X[cat_feat].cat.add_categories("Missing").fillna("Missing")
        
        X[cat_cols] = X[cat_cols].astype('str', copy=False)

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    end = time.time()
    pred_runtime = round(end-start, 2)

    return y_pred, y_proba, pred_runtime

def get_binary_results(y_true, y_pred, y_pred_proba, metric_df, metrics, split):
    for metric in metrics:
        score, _ = classif_metrics(metric, y_true, y_pred, y_pred_proba)
        metric_df.loc[0, f"{split} {metric}"] = score

    return metric_df

def get_multi_results(y_true, y_pred, y_pred_proba, metric_df, metrics, split, class_names):
    for metric in metrics:
        score, class_scores = classif_metrics(metric, y_true, y_pred, y_pred_proba, class_names)
        metric_df.loc[0, f"{split} {metric}"] = score # Overall
        if(class_scores is not None):
            for i, class_name in enumerate(class_names):
                metric_df.loc[0, f"{split} {metric}-{class_name}"] = class_scores[i] # Each Class
    
    return metric_df