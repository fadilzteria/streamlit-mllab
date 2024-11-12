import time
import numpy as np
from sklearn.preprocessing import label_binarize

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

def define_models(model_names, params):
    methods = {}
    for model_name in model_names:
        model_key = "_".join(model_name.lower().split(" "))

        # Linear Model
        if(model_name=="Logistic Regression"):
            methods[model_name] = LogisticRegression(random_state=42)
        
        # Discriminant Analysis
        elif(model_name=="Linear Discriminant Analysis"):
            methods[model_name] = LinearDiscriminantAnalysis()

        # Naive Bayes
        elif(model_name=="Bernoulli Bayes"):
            methods[model_name] = BernoulliNB()
        elif(model_name=="Gaussian Bayes"):
            methods[model_name] = GaussianNB()

        # SVM
        elif(model_name=="SVC"):
            methods[model_name] = SVC(probability=True)
        elif(model_name=="Linear SVC"):
            methods[model_name] = LinearSVC(random_state=42)

        # Neighbors
        elif(model_name=="KNN"):
            methods[model_name] = KNeighborsClassifier()

        # Tree
        elif(model_name=="Decision Tree"):
            max_depth = params[f"{model_key}_params_max_depth"]
            methods[model_name] = DecisionTreeClassifier(max_depth=max_depth, random_state=42, class_weight='balanced')
        elif(model_name=="Extra Trees"):
            max_depth = params[f"{model_key}_params_max_depth"]
            methods[model_name] = ExtraTreesClassifier(max_depth=max_depth, random_state=42, class_weight='balanced')
        elif(model_name=="Random Forest"):
            max_depth = params[f"{model_key}_params_max_depth"]            
            methods[model_name] = RandomForestClassifier(max_depth=max_depth, random_state=42, class_weight='balanced')
        elif(model_name=="AdaBoost"):
            methods[model_name] = AdaBoostClassifier(random_state=42)
        elif(model_name=="Gradient Boosting"):
            max_depth = params[f"{model_key}_params_max_depth"]            
            methods[model_name] = GradientBoostingClassifier(max_depth=max_depth, random_state=42)

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
    y_proba = model.predict_proba(X)

    end = time.time()
    pred_runtime = round(end-start, 2)

    return y_pred, y_proba, pred_runtime

def get_binary_results(y_true, y_pred, y_pred_proba, metric_df, metrics, split):
    for metric in metrics:
        if(metric=="Accuracy"):
            score = accuracy_score(y_true, y_pred) # Accuracy
        elif(metric=="Precision"):
            score = precision_score(y_true, y_pred) # Precision
        elif(metric=="Recall"):
            score = recall_score(y_true, y_pred) # Recall
        elif(metric=="F1 Score"):
            score = f1_score(y_true, y_pred) # F1 Score
        elif(metric=="ROC AUC"):
            score = roc_auc_score(y_true, y_pred_proba[:, 1]) # ROC AUC
        elif(metric=="Avg Precision"):
            score = average_precision_score(y_true, y_pred_proba[:, 1]) # Avg Precision
        metric_df.loc[0, f"{split} {metric}"] = score

    return metric_df

def get_multi_results(y_true, y_pred, y_pred_proba, metric_df, metrics, class_names, split):
    # Class
    y_true_classes = np.unique(y_true)
    y_true_onehot = label_binarize(y_true, classes=y_true_classes)

    for metric in metrics:
        if(metric=="Accuracy"):
            score = accuracy_score(y_true, y_pred) # Accuracy
            metric_df.loc[0, f"{split} {metric}"] = score
        else:
            if(metric=="Precision"): # Precision
                score = precision_score(y_true, y_pred, average="macro") 
                class_scores = precision_score(y_true, y_pred, average=None)
            elif(metric=="Recall"): # Recall
                score = recall_score(y_true, y_pred, average="macro") 
                class_scores = recall_score(y_true, y_pred, average=None)
            elif(metric=="F1 Score"): # F1 Score
                score = f1_score(y_true, y_pred, average="macro") 
                class_scores = f1_score(y_true, y_pred, average=None)
            elif(metric=="ROC AUC"): # ROC AUC
                score = roc_auc_score(y_true, y_pred_proba, multi_class='ovr') 
                class_scores = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average=None)
            elif(metric=="Avg Precision"): # Avg Precision
                score = average_precision_score(y_true_onehot, y_pred_proba)
                class_scores = average_precision_score(y_true_onehot, y_pred_proba, average=None)

            metric_df.loc[0, f"{split} {metric}"] = score # Overall
            for i, class_name in enumerate(class_names):
                metric_df.loc[0, f"{split} {metric}-{class_name}"] = class_scores[i] # Each Class

    return metric_df