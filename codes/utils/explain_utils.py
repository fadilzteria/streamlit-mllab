import os
import json
import pickle
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import shap
import streamlit as st

from codes.utils import train_utils, features

# Explain the Model
# @st.cache_data()
def explain_model(config, split="Train"):
    # Read Train Config
    exp_path = os.path.join("experiments", config["exp_name"])
    config_filepath = os.path.join(exp_path, "config.json")
    with open(config_filepath, 'r') as file:
        train_config = json.load(file)

    # Read Cleaned Train Dataset
    dp_path = os.path.join("datasets/cleaned_dataset", train_config["dp_name"])
    filename = "cleaned_train.parquet" if(split=="Train") else "cleaned_test.parquet"
    df_path = os.path.join(dp_path, filename)
    df = pd.read_parquet(df_path)

    # Read FE Sets and Pipeline
    fold = config["fold_modelx"]
    fold_path = os.path.join(exp_path, f"fold_{fold}")
    fe_sets_filepath = os.path.join(fold_path, f"fe_sets_fold_{fold}.json")
    with open(fe_sets_filepath, 'r') as file:
        fe_sets = json.load(file)

    fe_pipeline_filepath = os.path.join(fold_path, f"fe_pipeline_fold_{fold}.pkl")
    with open(fe_pipeline_filepath, 'rb') as file:
        fe_pipeline = pickle.load(file)

    # Cross Validation and Feature Engineering 
    if(split=="Train"):
        target_col = train_config["target"]
        df = train_utils.cross_validation(df, train_config["folds"], target_col)
        df = df[df['fold'] == fold].reset_index(drop=True)
    
    if(split in ["Train", "Valid"]):
        X_data, _, _ = features.feature_engineering(
            df, config=train_config, fe_sets=fe_sets, split="Valid", pipeline=fe_pipeline, 
        )
    else:
        X_data, _ = features.feature_engineering(
            df, config=train_config, fe_sets=fe_sets, split="Test", pipeline=fe_pipeline, 
        )

    # Model Choice
    model_path = os.path.join(fold_path, "models")
    model_filepath = os.path.join(model_path, f"{config['tree_model']}_fold_{fold}.model")
    with open(model_filepath, 'rb') as file:
        ex_model = pickle.load(file)

    # SHAP Explainer
    explainer = shap.TreeExplainer(ex_model)
    explanation = explainer(X_data)
    shap_values = explanation.values

    # Unique Targets
    if(train_config["ml_task"]=="Classification"):
        unique_targets = fe_pipeline["Class Names"]
    else:
        unique_targets = [1]

    return ex_model, explainer, explanation, shap_values, X_data, unique_targets

# region
# Bar Plot
def show_shap_bar_plot(explanation, targets, max_display=20):
    if(len(targets) > 2):
        fig, axs = plt.subplots(nrows=len(targets), ncols=1, figsize=(15, len(targets)*6))
        for i, target in enumerate(targets):
            shap.plots.bar(explanation[:, :, i], ax=axs[i], show=False)
            axs[i].set_title(target)
    else:
        if(len(explanation.shape) > 2):
            explanation = explanation[:, :, 1]
        fig, axs = plt.subplots(figsize=(15, 6))
        shap.plots.bar(explanation, max_display=max_display, ax=axs, show=False)
    st.pyplot(fig)

# Beeswarm Plot
def show_shap_beeswarm_plot(explanation, targets, max_display=20):
    if(len(targets) > 2):
        fig, axs = plt.subplots(nrows=len(targets), ncols=1, figsize=(15, len(targets)*6))
        for i, target in enumerate(targets):
            plt.subplot(len(targets), 1, i+1)
            shap.plots.beeswarm(explanation[:, :, i], show=False, plot_size=None)
            axs[i].set_title(target)
    else:
        if(len(explanation.shape) > 2):
            explanation = explanation[:, :, 1]
        fig, axs = plt.subplots(figsize=(15, 6))
        axs = shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    st.pyplot(fig)

# Feature Importance Dataframe with Threshold
def extract_feat_importance(shap_values, X_data, imp_thres):
    feature_names = X_data.columns
    shap_df = pd.DataFrame(shap_values, columns=feature_names)

    mean_vals = np.abs(shap_df.values).mean(0)
    min_vals, max_vals = np.min(shap_df.values, axis=0), np.max(shap_df.values, axis=0)

    shap_importance_df = pd.DataFrame(
        list(zip(feature_names, mean_vals, min_vals, max_vals)),
        columns=["Feature", "Importance", "Min Shap Values", "Max Shap Values"]
    )
    shap_importance_df = shap_importance_df.sort_values(by="Importance", ascending=False).reset_index(drop=True)
    shap_importance_df = shap_importance_df[shap_importance_df["Importance"]>=imp_thres]

    return shap_importance_df

# Partial Dependence (PD) and Individual Conditional Expectation (ICE) Plots with Sklearn
def show_sklearn_pd_ice_plots(ex_model, X_data, features, cat_features, ice=False):
    if(ice):
        features = [x for x in features if x not in cat_features]

    # Create Subplots
    nrows, ncols = ((len(features)-1)//3)+1, 3
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, nrows*4))

    # Plots
    for i, feature in enumerate(features):
        ax_temp = axs[i] if nrows==1 else axs[i//ncols, i%ncols]
        if(ice):
            PartialDependenceDisplay.from_estimator(
                ex_model, X_data, [feature], 
                random_state=42, ax=ax_temp, grid_resolution=50, kind="individual"
            )
        else:
            PartialDependenceDisplay.from_estimator(
                ex_model, X_data, [feature], categorical_features=cat_features,
                random_state=42, ax=ax_temp, grid_resolution=50
            )

    if(i < (nrows*ncols)):
        for j in range(i+1, (nrows*ncols)):
            ax_temp = axs[j] if nrows==1 else axs[j//ncols, j%ncols]
            ax_temp.axis("off")

    if(ice):
        plt.suptitle("Individual Conditional Expectation Plots with scikit-learn")
    else:
        plt.suptitle("Partial Dependence Plots with scikit-learn")
    plt.tight_layout()
    st.pyplot(fig)

# Show PD/ICE Plots
def show_pd_plots(ex_model, shap_values, X_data, targets, imp_thres, ice):
    # Numerical to Binary
    num_features = X_data.select_dtypes(include=['int', 'float'])
    bool_features = [feat for feat in num_features if X_data[feat].nunique()==2]
    X_data[bool_features] = X_data[bool_features].astype(bool)

    if(len(targets) > 2):
        for i, target in enumerate(targets):
            # Feature Importance
            shap_importance_df = extract_feat_importance(shap_values[:, :, i], X_data, imp_thres)
            if(len(shap_importance_df) == 0):
                st.warning("No Feature Importance")
                continue

            st.write("Target:", target)
            st.dataframe(shap_importance_df)

            # Features
            features = list(shap_importance_df["Feature"])
            cat_features = X_data.select_dtypes(exclude=['int', 'float']).columns.tolist()

            # PD Plots
            show_sklearn_pd_ice_plots(ex_model, X_data, features, cat_features, ice)    
    else:
        if(len(shap_values.shape) > 2):
            shap_values = shap_values[:, :, 1]

        # Feature Importance
        shap_importance_df = extract_feat_importance(shap_values, X_data, imp_thres)
        if(len(shap_importance_df) > 0):
            st.dataframe(shap_importance_df)

            # Features
            features = list(shap_importance_df["Feature"])
            cat_features = X_data.select_dtypes(exclude=['int', 'float']).columns.tolist()

            # PD Plots
            show_sklearn_pd_ice_plots(ex_model, X_data, features, cat_features, ice)    
        else:
            st.warning("No Feature Importance")
# endregion

# Easy-Difficult Samples
def show_easy_difficult_samples(df, model_names, target_column, str_pred, spec_model_name, target_name=""):
    full_model_rank = "Full_Model_Worst"
    df[full_model_rank] = 0

    for model_name in model_names:
        pred_name = f"{model_name}_{target_column}{target_name}_{str_pred}"
        model_rank = f"{model_name}_Worst{target_name}"
        if(str_pred=="proba"): # Classification
            df[model_rank] = df[pred_name].rank(ascending=False)
        else: # Regression
            model_diff = f"{model_name}_Diff"
            df[model_diff] = abs(df[pred_name] - df[target_column])
            df[model_rank] = df[model_diff].rank(ascending=True)
        df[full_model_rank] += df[model_rank]

    df[full_model_rank] /= len(model_names)
    sorted_rank = full_model_rank if(spec_model_name=="") else f"{spec_model_name}_Worst{target_name}"

    st.write("Easy Samples")
    df = df.sort_values(by=sorted_rank, ascending=True).reset_index(drop=True)
    st.dataframe(df, height=220)

    st.write("Difficult Samples")
    df = df.sort_values(by=sorted_rank, ascending=False).reset_index(drop=True)
    st.dataframe(df, height=220)

def extract_easy_difficult_samples(train_config, oof_df, spec_model_name="", fold=0, split="Train"):
    # Filter Dataframe
    if(split=="Train"):
        oof_df = oof_df[oof_df["fold"]==fold].reset_index(drop=True)
    oof_df["index"] = oof_df.index.values.tolist()    

    # Features
    target_column = train_config["target"]
    features = [train_config["id"], "index", target_column]
    str_pred = "proba" if(train_config["ml_task"]=="Classification") else "pred"
    pred_columns = [col for col in oof_df.columns if str_pred in col]
    features_with_pred = features.copy()
    features_with_pred.extend(pred_columns)
    filtered_df = copy.deepcopy(oof_df[features_with_pred])

    # Model Names
    model_names = [col.split("_")[0] for col in oof_df.columns if "pred" in col]
    if(split=="Valid"):
        model_names = [model_name+f"_{fold}" for model_name in model_names]
    
    if(train_config["ml_task"]=="Classification"):
        target_names = oof_df[target_column].unique().tolist()
        for target_name in target_names:
            target_df = filtered_df[filtered_df[target_column]==target_name].reset_index(drop=True)
            tpred_columns = [col for col in pred_columns if str(target_name) in col]
            if(split=="Valid"):
                tpred_columns = [col for col in tpred_columns if col.split("_")[1] == str(fold)]
            features_with_tpred = features.copy()
            features_with_tpred.extend(tpred_columns)
            target_df = target_df[features_with_tpred]

            st.write("Target:", target_name)

            show_easy_difficult_samples(
                target_df, model_names, target_column, str_pred, spec_model_name=spec_model_name, target_name=f"_{target_name}"
            )
    else:
        show_easy_difficult_samples(filtered_df, model_names, target_column, str_pred, spec_model_name)

# Local Force Plot
def show_shap_local_force_plot(explainer, shap_values, X_data, targets, idx):
    if(len(targets) > 2):
        for i, target in enumerate(targets):
            fig = shap.plots.force(
                explainer.expected_value[i], shap_values[idx, :, i], X_data.iloc[idx, :].round(2),
                feature_names=X_data.columns, matplotlib=True, show=False, figsize=(15, 3)
            )
            st.write(target)
            st.pyplot(fig)
    else:
        if(len(shap_values.shape) > 2):
            expected_value = explainer.expected_value[1]
            shap_values = shap_values[:, :, 1]
        else:
            expected_value = explainer.expected_value

        fig = shap.plots.force(
            expected_value, shap_values[idx], X_data.iloc[idx].round(2),
            feature_names=X_data.columns, matplotlib=True, show=False, figsize=(15, 3)
        )
        st.pyplot(fig)

# Local Bar Plot
def show_shap_local_bar_plot(explanation, targets, idx):
    if(len(targets) > 2):
        fig, axs = plt.subplots(nrows=len(targets), ncols=1, figsize=(15, len(targets)*8))
        for i, target in enumerate(targets):
            plt.subplot(len(targets), 1, i+1)
            shap.plots.bar(explanation[idx, :, i], max_display=15, show=False, ax=axs[i])
            axs[i].set_title(target)
        st.pyplot(fig)
    else:
        fig, axs = plt.subplots(figsize=(15, 8))
        if(len(explanation.shape) > 2):
            explanation = explanation[:, :, 1]
        shap.plots.bar(explanation[idx], max_display=15, show=False, ax=axs)
        st.pyplot(fig)

# Local Explainability
def show_local_explainability(explainer, explanation, shap_values, X_data, targets, idx):
    # # Local Force Plot
    st.subheader("Local Force Plot")
    show_shap_local_force_plot(explainer, shap_values, X_data, targets, idx)

    # Local Bar Plot
    st.subheader("Local Bar Plot")
    show_shap_local_bar_plot(explanation, targets, idx)