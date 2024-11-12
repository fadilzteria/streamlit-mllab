import copy
import os
import json
import pickle
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import streamlit as st
from stqdm import stqdm

from codes.utils import classification as classif, regression as regress, features

def cross_validation(train_df, folds, target_column):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    if(train_df[target_column].dtypes in ["int", "float"]):
        train_df["binned"] = pd.qcut(train_df[target_column], q=10).astype('str')
        for fold, (_, valid_index) in enumerate(skf.split(train_df, train_df["binned"])):
            train_df.loc[valid_index, "fold"] = int(fold)
        train_df = train_df.drop("binned", axis=1)
    else:
        for fold, (_, valid_index) in enumerate(skf.split(train_df, train_df[target_column])):
            train_df.loc[valid_index, "fold"] = int(fold)

    return train_df

def training_loop(config, df, fe_sets, methods, metrics, fold):
    print(f"========== Fold: {fold} Training ==========")
    fold_path = os.path.join("experiments", config["exp_name"], f"fold_{fold}")

    # ======== SPLIT ==========
    train_folds = df[df['fold'] != fold].reset_index(drop=True)
    valid_folds = df[df['fold'] == fold].reset_index(drop=True)
    oof_df = copy.deepcopy(valid_folds[[config["id"], config["target"], "fold"]])

    # ======== FEATURE ENGINEERING ==========
    X_train, y_train, _, _, fe_pipeline = features.feature_engineering(
        train_folds, config=config, fe_sets=fe_sets, split="Train"
    )
    X_valid, y_valid, _ = features.feature_engineering(
        valid_folds, config=config, fe_sets=fe_sets, split="Valid", pipeline=fe_pipeline, 
    )

    print(f"Training Data Shape: {X_train.shape}")
    print(f"Validation Data Shape: {X_valid.shape}")

    if(config["ml_task"]=="Classification"):
        class_names = list(train_folds[config["target"]].unique())
        class_names = [bool(c) if isinstance(c, np.bool_) else c for c in class_names]
        fe_pipeline["Class Names"] = class_names
        pred_mapping = dict(zip(range(len(class_names)), class_names))
        print(f"Classes: {class_names}")

    # Save Feature Engineering Sets and Pipeline
    fe_sets_filepath = os.path.join(fold_path, f"fe_sets_fold_{fold}.json")
    with open(fe_sets_filepath, "w") as f:
        json.dump(fe_sets, f)

    fe_pipeline_filepath = os.path.join(fold_path, f"fe_pipeline_fold_{fold}.pkl")
    pickle.dump(fe_pipeline, open(fe_pipeline_filepath, 'wb'))

    # ======== TRAINING ==========
    full_metric_df = pd.DataFrame()
    best_method = ""
    best_metric = 0.0 if config["best_value"]=="Maximize" else np.inf

    model_path = os.path.join(fold_path, "models")
    os.mkdir(model_path)

    for i, (model_name, model) in stqdm(enumerate(methods.items()), desc="Models"):
        print(model_name)

        # Training
        if(config["ml_task"]=="Classification"):
            model, train_runtime = classif.train_function(model, X_train, y_train)
        else:
            model, train_runtime = regress.train_function(model, X_train, y_train)

        # Prediction
        if(config["ml_task"]=="Classification"):
            y_train_pred, y_train_proba, pred_runtime_1 = classif.predict_function(model, X_train)
            y_valid_pred, y_valid_proba, pred_runtime_2 = classif.predict_function(model, X_valid)
        else:
            y_train_pred, pred_runtime_1 = regress.predict_function(model, X_train)
            y_valid_pred, pred_runtime_2 = regress.predict_function(model, X_valid)
        pred_runtime = pred_runtime_1 + pred_runtime_2

        # Assign Prediction to Valid Folds
        oof_df[f"{model_name}_{config['target']}_pred"] = y_valid_pred
        if(config["ml_task"]=="Classification"):
            if(oof_df[config['target']].dtypes=="object"):
                oof_df[f"{model_name}_{config['target']}_pred"] = oof_df[f"{model_name}_{config['target']}_pred"].map(pred_mapping)
            for i, class_name in enumerate(class_names):
                oof_df[f"{model_name}_{config['target']}_{class_name}_proba"] = y_valid_proba[:, i]

        # Evaluation
        metric_df = pd.DataFrame({f"Fold": fold, "Model": model_name}, index=[0])
        if(config["ml_task"]=="Classification"):
            if(len(class_names)==2): # Binary
                metric_df = classif.get_binary_results(y_train, y_train_pred, y_train_proba, metric_df, metrics, split="Train")
                metric_df = classif.get_binary_results(y_valid, y_valid_pred, y_valid_proba, metric_df, metrics, split="Valid")
            else: # Multi-class
                metric_df = classif.get_multi_results(y_train, y_train_pred, y_train_proba, metric_df, metrics, class_names, split="Train")
                metric_df = classif.get_multi_results(y_valid, y_valid_pred, y_valid_proba, metric_df, metrics, class_names, split="Valid")
        else:
            n, p = X_train.shape[0], X_train.shape[1]
            metric_df = regress.get_results(y_train, y_train_pred, metric_df, metrics, split="Train", n=n, p=p)
            metric_df = regress.get_results(y_valid, y_valid_pred, metric_df, metrics, split="Valid", n=n, p=p)

        metric_df.loc[0, "Training Runtime"] = train_runtime
        metric_df.loc[0, "Prediction Runtime"] = pred_runtime

        full_metric_df = pd.concat([full_metric_df, metric_df], ignore_index=True)

        # Print Results
        for split in ["Train", "Valid"]:
            print(f"{split} Metrics", end="")
            for metric in metrics:
                print(f", {metric}: {metric_df.loc[0, f'{split} {metric}']:.4f}", end="")
            print("")
        print(f"Training Runtime: {metric_df.loc[0, 'Training Runtime']}s, Prediction Runtime: {metric_df.loc[0, 'Prediction Runtime']}s\n")

        # Compare with Temporary Target Metrics
        if(config["best_value"]=="Maximize"):
            if(metric_df.loc[0, f"Valid {config['best_metric']}"] > best_metric):
                best_metric = metric_df.loc[0, f"Valid {config['best_metric']}"]
                best_method = model_name
        else:
            if(metric_df.loc[0, f"Valid {config['best_metric']}"] < best_metric):
                best_metric = metric_df.loc[0, f"Valid {config['best_metric']}"]
                best_method = model_name

        # Save Model
        model_filepath = os.path.join(model_path, f"{model_name}_fold_{fold}.model")
        pickle.dump(model, open(model_filepath, 'wb'))

    print(f"Best Model: {best_method} | ROC AUC: {best_metric}\n")

    return oof_df, full_metric_df

def training_and_validation(config, train_df, fe_sets, methods, params, metrics):
    # Experiment Folder
    exp_path = os.path.join("experiments", config["exp_name"])
    if(os.path.exists(exp_path)):
        shutil.rmtree(exp_path)
    os.mkdir(exp_path)

    # Save Config
    config_filepath = os.path.join(exp_path, "config.json")
    with open(config_filepath, "w") as f:
        json.dump(config, f)

    train_df = cross_validation(train_df, config["folds"], config["target"]) # Cross Validation

    # Define Methods with Its Params
    if(config["ml_task"]=="Classification"): 
        methods = classif.define_models(methods, params)
    else:
        methods = regress.define_models(methods, params)

    # Variables
    oof_df = pd.DataFrame()
    all_metric_df = pd.DataFrame()

    folds = train_df["fold"].nunique()
    for fold in stqdm(range(folds), desc="Folds"):
        fold_path = os.path.join(exp_path, f"fold_{fold}")
        os.mkdir(fold_path)
        
        temp_methods = copy.deepcopy(methods)

        _oof_df, _metric_df = training_loop(config, train_df, fe_sets, temp_methods, metrics, fold)

        oof_df = pd.concat([oof_df, _oof_df])
        all_metric_df = pd.concat([all_metric_df, _metric_df])

    oof_df = oof_df.sort_values(by=config["id"]).reset_index(drop=True)
    oof_filepath = os.path.join(exp_path, "oof_df.csv")
    oof_df.to_csv(oof_filepath, index=False)

    all_metric_df = all_metric_df.reset_index(drop=True)
    all_metric_filepath = os.path.join(exp_path, "metric_df.csv")
    all_metric_df.to_csv(all_metric_filepath, index=False)

    st.success(f"Your training has been successfully processed and saved in the experiments folder as {config['exp_name']}")