import os
import json
import copy
import pickle
import shutil
import numpy as np
import pandas as pd
import streamlit as st
from stqdm import stqdm

from catboost import Pool

from codes.utils import classification as classif, regression as regress, features

def extract_model_names(config):
    methods = config["methods"]
    n_models = config["n_models"]
    model_names = []
    for model_name in methods:
        model_key = "_".join(model_name.lower().split(" "))
        for n in range(1, n_models[f"{model_key}_n_models"]+1):
            model_n_name = f"{model_name} {n}"
            model_names.append(model_n_name)
    
    return model_names

def inference(test_config, train_config, test_df, methods):
    pred_df = pd.DataFrame()
    pred_df[train_config["id"]] = test_df[train_config["id"]]

    # Each Fold
    exp_path = os.path.join("experiments", test_config["exp_name"])
    for fold in stqdm(test_config["folds"]):
        fold_path = os.path.join(exp_path, f"fold_{fold}")

        # Feature Engineering
        fe_sets_filepath = os.path.join(fold_path, f"fe_sets_fold_{fold}.json")
        with open(fe_sets_filepath, 'r') as file:
            fe_sets = json.load(file)
        
        fe_pipeline_filepath = os.path.join(fold_path, f"fe_pipeline_fold_{fold}.pkl")
        with open(fe_pipeline_filepath, 'rb') as file:
            fe_pipeline = pickle.load(file)
        
        if(train_config["target"] in test_df.columns):
            X_test, _, _ = features.feature_engineering(
                test_df, config=train_config, fe_sets=fe_sets, split="Valid", pipeline=fe_pipeline, 
            )
        else:
            X_test, _ = features.feature_engineering(
                test_df, config=train_config, fe_sets=fe_sets, split="Test", pipeline=fe_pipeline, 
            )

        if(train_config["ml_task"]=="Classification"): 
            class_names = fe_pipeline["Class Names"]
            pred_mapping = dict(zip(range(len(class_names)), class_names))
        
        # Each Model
        model_path = os.path.join(fold_path, "models")
        for model_name in stqdm(methods):
            model_filepath = os.path.join(model_path, f"{model_name}_fold_{fold}.model")
            with open(model_filepath, 'rb') as file:
                model = pickle.load(file)
                
            if("CatBoost" in model_name):
                cat_cols = list(X_test.select_dtypes(include=['category']).columns.values)
                for cat_feat in cat_cols:
                    if(X_test[cat_feat].isnull().values.any()):
                        X_test[cat_feat] = X_test[cat_feat].cat.add_categories("Missing").fillna("Missing")

                cat_cols = list(X_test.select_dtypes(include=['category']).columns.values)
                # cat_cols = X_test.columns.tolist()

                X_test[cat_cols] = X_test[cat_cols].astype('str', copy=False)
                test_data = Pool(data=X_test, cat_features=cat_cols)

            if(train_config["ml_task"]=="Classification"): 
                target_name = f"{model_name}_{fold}_{train_config['target']}"
                if("CatBoost" in model_name):
                    y_pred = model.predict_proba(test_data)
                elif(model_name=="Linear SVC"):
                    y_pred = model.predict(X_test)
                else:
                    y_pred = model.predict_proba(X_test)

                for i, class_name in enumerate(class_names):
                    pred_df[f"{target_name}_{class_name}_proba"] = y_pred[:, i]
                
                pred_df[f"{target_name}_pred"] = np.argmax(y_pred, axis=1)
                pred_df[f"{target_name}_pred"] = pred_df[f"{target_name}_pred"].map(pred_mapping)
            
            else:
                if("CatBoost" in model_name):
                    y_pred = model.predict(test_data)
                else:
                    y_pred = model.predict(X_test)
                pred_df[f"{model_name}_{fold}_{train_config['target']}_pred"] = y_pred

    return pred_df

def ensembling(test_config, train_config, test_df, pred_df, model_name=""):
    ensembled_df = pd.DataFrame()
    ensembled_df[train_config["id"]] = test_df[train_config["id"]]

    if(model_name=="Ensembled"):
        pred_columns = pred_df.columns[1:].tolist()
    else:
        pred_columns = [col for col in pred_df.columns if model_name in col]
    
    target_name = f"{model_name}_{train_config['target']}"
    hard_target_name = target_name + "_pred"

    if(train_config["ml_task"]=="Classification"): 
        # Class Names
        class_names = [col.split("_")[-2] for col in pred_columns if "proba" in col]
        class_names = sorted(set(class_names), key=class_names.index)

        # Soft Classes
        for i, class_name in enumerate(class_names):
            class_columns = [col for col in pred_columns if class_name in col]
            class_target_name = target_name + "_" + class_name
            ensembled_df[class_target_name] = np.mean(pred_df.loc[:, class_columns], axis=1)

        # Hard Classes
        ensembled_df[hard_target_name] = np.argmax(ensembled_df.iloc[:, 1:len(class_names)+1], axis=1)
        ensembled_df[hard_target_name] = ensembled_df[hard_target_name].apply(lambda x: class_names[x])
        uniques = ensembled_df[hard_target_name].unique()
        if(len(uniques)==2 and (np.sort(uniques) == ['False', 'True']).all()):
            ensembled_df[hard_target_name] = (ensembled_df[hard_target_name]=='True')
    
    else:
        ensembled_df[hard_target_name] = np.mean(pred_df.loc[:, pred_columns], axis=1)
    
    return ensembled_df

def full_ensembling(test_config, train_config, test_df, pred_df):
    full_ensembled_df = pd.DataFrame()
    full_ensembled_df[train_config["id"]] = test_df[train_config["id"]]

    methods = copy.deepcopy(test_config["methods"])
    methods.append("Ensembled")
    for model_name in methods:
        ensembled_df = ensembling(test_config, train_config, test_df, pred_df, model_name=model_name)
        full_ensembled_df = full_ensembled_df.merge(ensembled_df, on=train_config["id"])

    if(train_config["target"] in test_df.columns):
        full_ensembled_df[f"{train_config['target']}_actual"] = test_df[train_config["target"]]

    base_test_path = os.path.join("experiments", train_config["exp_name"], "tests")
    test_path = os.path.join(base_test_path, test_config["test_name"])

    ensembled_filepath = os.path.join(test_path, "ensembled_df.parquet")
    full_ensembled_df.to_parquet(ensembled_filepath)
    
    return full_ensembled_df

def eval_testing(test_config, train_config, ensembled_df):
    metric_list = train_config["metrics"]
    full_metric_df = pd.DataFrame()
    actual_target_name = f"{train_config['target']}_actual"

    methods = copy.deepcopy(test_config["methods"])
    methods.append("Ensembled")

    for model_name in methods:
        model_target_name = f"{model_name}_{train_config['target']}_pred"
        model_metric_df = pd.DataFrame({"Model": model_name}, index=[0])

        y_true = ensembled_df[actual_target_name]        
        y_pred = ensembled_df[model_target_name]
        
        if(train_config["ml_task"]=="Classification"):
            model_proba_names = [col for col in ensembled_df.columns if model_name in col and "pred" not in col]
            y_pred_proba = np.array(ensembled_df[model_proba_names])

            if(ensembled_df[actual_target_name].nunique() == 2):
                model_metric_df = classif.get_binary_results(
                    y_true, y_pred, y_pred_proba, model_metric_df, metric_list, split=""
                )
            else:
                class_names = ensembled_df[actual_target_name].unique().tolist()
                model_metric_df = classif.get_multi_results(
                    y_true, y_pred, y_pred_proba, model_metric_df, metric_list, split="", class_names=class_names
                )
        else:
            n, p = ensembled_df.shape[0], ensembled_df.shape[1]
            model_metric_df = regress.get_results(
                y_true, y_pred, model_metric_df, metric_list, split="", n=n, p=p
            )
        
        full_metric_df = pd.concat([full_metric_df, model_metric_df], ignore_index=True)

    base_test_path = os.path.join("experiments", train_config["exp_name"], "tests")
    test_path = os.path.join(base_test_path, test_config["test_name"])

    metric_filepath = os.path.join(test_path, "metric_df.parquet")
    full_metric_df.to_parquet(metric_filepath)
    
    return full_metric_df