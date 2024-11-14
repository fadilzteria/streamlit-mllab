import os
import json
import copy
import pandas as pd
import streamlit as st

from codes.utils import eval_utils

# ==================================================================================================
# EVALUATION
# ==================================================================================================
@st.fragment()
def all_metrics(config, metric_df):
    folds = list(range(config["folds"]))
    fold_options = ["All", *folds]
    st.selectbox(label="Folds for All Metrics", options=fold_options, key="fold_all_metrics", index=0)
    eval_utils.show_metrics(config, metric_df, fold=st.session_state["fold_all_metrics"])

@st.fragment()
def class_wise_metrics(config, metric_df):
    folds = list(range(config["folds"]))
    fold_options = ["All", *folds]
    st.selectbox(label="Folds for Class-wise Metrics", options=fold_options, key="fold_cw_metrics", index=0)
    eval_utils.get_class_metrics(config, metric_df, fold=st.session_state["fold_cw_metrics"])

@st.fragment()
def training_runtime(config, metric_df):
    folds = list(range(config["folds"]))
    fold_options = ["All", *folds]
    st.selectbox(label="Folds for Training Runtime", options=fold_options, key="fold_runtime", index=0)
    eval_utils.show_runtime(metric_df, fold=st.session_state["fold_runtime"])

def model_evaluation():
    exp_list = os.listdir("experiments")
    if(len(exp_list)>0):
        # Exp Name for Evaluation
        st.selectbox(label="Experiment Name", options=exp_list, key="exp_name_eval", index=0) 
        exp_path = os.path.join("experiments", st.session_state["exp_name_eval"])

        # Read Necessary Files
        config_filepath = os.path.join(exp_path, "config.json")
        with open(config_filepath, 'r') as file:
            config = json.load(file)

        oof_filepath = os.path.join(exp_path, "oof_df.parquet")
        oof_df = pd.read_parquet(oof_filepath)

        metric_filepath = os.path.join(exp_path, "metric_df.parquet")
        metric_df = pd.read_parquet(metric_filepath)

        # All Metrics
        st.header("All Metrics", divider="orange")
        all_metrics(config, metric_df)

        # Class-wise Metrics
        if(config["ml_task"]=="Classification" and len(oof_df[config["target"]].unique()) > 2):
            st.header("Class-wise Metrics", divider="orange")
            class_wise_metrics(config, metric_df)

        # Confusion Matrix
        if(config["ml_task"]=="Classification"):
            st.header("Confusion Matrix", divider="orange")
            eval_utils.show_confusion_matrix(config, oof_df)

        # Category-based Matrices
        # ...

        # Training Runtime
        st.header("Training Runtime", divider="orange")
        training_runtime(config, metric_df)

        # Regression Diagnostics
        # ...

        # Predicted Distributions
        # ...

        # Easy and Hard Samples
        # ...
        
    else:
        st.warning("You need to train your models to explore model evaluation")

# ==================================================================================================
# MAIN
# ==================================================================================================
# Title
st.title("Evaluation")

# Evaluation
model_evaluation()

# # Training
# if("cleaned_dataset" in st.session_state):
#     model_training()

# exp_path = os.path.join("experiments", st.session_state["exp_name"])
# if(os.path.exists(exp_path)):
#     show_training_dataframe(exp_path)