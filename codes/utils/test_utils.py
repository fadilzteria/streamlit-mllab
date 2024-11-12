import os
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from stqdm import stqdm

from codes.utils import classification as classif, regression as regress, features

def inference(test_config, test_df, methods):
    # Config
    exp_path = os.path.join("experiments", test_config["exp_name"])
    config_filepath = os.path.join(exp_path, "config.json")
    with open(config_filepath, 'r') as file:
        train_config = json.load(file)

    pred_df = pd.DataFrame()
    pred_df[train_config["id"]] = test_df[train_config["id"]]

    # Each Fold
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
        
        # Each Model
        model_path = os.path.join(fold_path, "models")
        for model_name in stqdm(methods):
            model_filepath = os.path.join(model_path, f"{model_name}_fold_{fold}.model")
            with open(model_filepath, 'rb') as file:
                model = pickle.load(file)

            if(train_config["ml_task"]=="Classification"): 
                if(model_name=="Linear SVC"):
                    y_pred = model.predict(X_test)
                else:
                    y_pred = model.predict_proba(X_test)

                for i, class_name in enumerate(class_names):
                    pred_df[f"{model_name}_{fold}_{train_config['target']}_{class_name}_proba"] = y_pred[:, i]
            
            else:
                y_pred = model.predict(X_test)
                pred_df[f"{model_name}_{fold}_{train_config['target']}_pred"] = y_pred

    return pred_df

# Ensembling
def ensembling(test_config, test_df, pred_df):
    # Config
    exp_path = os.path.join("experiments", test_config["exp_name"])
    config_filepath = os.path.join(exp_path, "config.json")
    with open(config_filepath, 'r') as file:
        train_config = json.load(file)

    pred_columns = pred_df.columns[1:].tolist()
    class_names = list(set([col.split("_")[-2] for col in pred_columns]))

    ensembled_df = pd.DataFrame()
    ensembled_df[train_config["id"]] = test_df[train_config["id"]]

    # Soft Classes
    for i, class_name in enumerate(class_names):
        class_columns = [col for col in pred_df.columns if class_name in col]
        ensembled_df[f"{train_config['target']}_{class_name}"] = np.mean(pred_df.loc[:, class_columns], axis=1)

    # Hard Classes
    ensembled_df[train_config['target']] = np.argmax(ensembled_df.iloc[:, 1:len(class_names)+1], axis=1)
    ensembled_df[train_config['target']] = ensembled_df[train_config['target']].apply(lambda x: class_names[x])

    st.success(f"Your testing has been successfully processed")

    return ensembled_df