import os
import json
import copy
import pandas as pd

import streamlit as st

from codes.utils import explain_utils, test_utils

# ==================================================================================================
# EXPLAIN THE MODEL
# ==================================================================================================
@st.fragment()
def fill_explainability_configuration():
    # Experiment Name
    exp_list = os.listdir("experiments")
    st.selectbox(label="Experiment Name", options=exp_list, key="exp_namex", index=0) 
    exp_path = os.path.join("experiments", st.session_state["exp_namex"])

    # Read Train Config
    config_filepath = os.path.join(exp_path, "config.json")
    with open(config_filepath, 'r') as file:
        config = json.load(file)

    col_1, col_2 = st.columns(2)
    with col_1:
        # Folds
        fold_options = list(range(config["folds"]))
        st.selectbox(label="Folds for Model Explainability", options=fold_options, key="fold_modelx", index=0)
    with col_2:
        # Models
        model_names = test_utils.extract_model_names(config)
        tree_model_names = []
        tree_names = [
            "Decision Tree", "Extra Trees", "Random Forest", "AdaBoost", "Gradient Boosting",
            "XGBoost", "LightGBM", "CatBoost"
        ]
        for tree_name in tree_names:
            temp_model_names = [model_name for model_name in model_names if tree_name in model_name]
            tree_model_names.extend(temp_model_names)
        st.selectbox(label="Tree Models", options=tree_model_names, key="tree_model", index=0)

    if(len(tree_model_names) == 0):
        st.warning("You need to have ensemble tree models such as XGBoost, LightGBM, and CatBoost from your experiment to explore model explainability")

def run_explainability():
    if(st.session_state["tree_model"] is not None):
        explain_config = {
            "exp_name": st.session_state["exp_namex"],
            "tree_model": st.session_state["tree_model"],
            "fold_modelx": st.session_state["fold_modelx"],
        }
        results = explain_utils.explain_model(explain_config)

        st.session_state["ex_model"], st.session_state["explainer"] = results[0:2]
        st.session_state["explanation"], st.session_state["shap_values"] = results[2:4]
        st.session_state["X_data"], st.session_state["unique_targets"] = results[4:]
    else:
        st.error("You need to have ensemble tree models such as XGBoost, LightGBM, and CatBoost from your experiment to explore model explainability")

def model_explainability():
    exp_list = os.listdir("experiments")
    if(len(exp_list)>0):
        st.header("Configuration", divider="orange")
        with st.container(border=True):
            fill_explainability_configuration()
            st.button("Explain the Model", on_click=run_explainability)
    else:
        st.warning("You need to train your models to explore model explainability")

# ==================================================================================================
# SHOW EXPLAINABILITY
# ==================================================================================================
@st.fragment()
def partial_dependence_plots():
    col_1, col_2 = st.columns(2)
    with col_1:
        st.slider(
            "Feature Importance Threshold",
            min_value=0.001, max_value=1.0, value=0.1, step=0.002, format="%.3f", key="imp_thres"
        )
    with col_2:
        st.toggle("Use ICE", value=False, key="bool_ice")

    explain_utils.show_pd_plots(
        st.session_state["ex_model"], st.session_state["shap_values"],
        copy.deepcopy(st.session_state["X_data"]), st.session_state["unique_targets"],
        st.session_state["imp_thres"], st.session_state["bool_ice"]
    )

@st.fragment()
def local_explainability():
    max_idx = len(st.session_state["X_data"]) - 1
    st.number_input("Index Dataframe", min_value=0, max_value=max_idx, value=0, key="idx")
    explain_utils.show_local_explainability(
        st.session_state["explainer"], st.session_state["explanation"], st.session_state["shap_values"], 
        st.session_state["X_data"], st.session_state["unique_targets"], idx=st.session_state["idx"]
    )

def show_explainability():
    # Global
    # ------------------------------------------------------
    st.header("Global", divider="orange")

    # Feature Importance Using Bar Plot
    st.subheader("Feature Importance using Bar Plot")
    explain_utils.show_shap_bar_plot(st.session_state["explanation"], st.session_state["unique_targets"])

    # Feature Importance Using Beeswarm Plot
    st.subheader("Feature Importance using Beeswarm Plot")
    explain_utils.show_shap_beeswarm_plot(st.session_state["explanation"], st.session_state["unique_targets"])

    # Partial Dependence Plot
    st.subheader("Partial Dependence Plots")
    partial_dependence_plots()

    # Easy-Difficult Samples
    # ------------------------------------------------------
    st.header("Easy-Difficult Samples", divider="orange")
    exp_path = os.path.join("experiments", st.session_state["exp_namex"])

    # Read Train Config
    config_filepath = os.path.join(exp_path, "config.json")
    with open(config_filepath, 'r') as file:
        config = json.load(file)

    # Read OOF Dataset
    oof_filepath = os.path.join(exp_path, "oof_df.parquet")
    oof_df = pd.read_parquet(oof_filepath)

    # Extraction
    explain_utils.extract_easy_difficult_samples(config, oof_df, st.session_state["tree_model"], st.session_state["fold_modelx"])

    # Local
    # ------------------------------------------------------
    st.header("Local", divider="orange")
    local_explainability()

# ==================================================================================================
# MAIN
# ==================================================================================================
# Title
st.title("Model Explainability") 

# Explain the Model
model_explainability()

# Show Explainability
if("explanation" in st.session_state): 
    show_explainability()