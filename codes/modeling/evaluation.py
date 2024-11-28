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
    st.selectbox(
        label="Folds for All Metrics", options=fold_options, key="fold_all_metrics", index=0
    )
    eval_utils.show_metrics(config, metric_df, fold=st.session_state["fold_all_metrics"])

@st.fragment()
def class_wise_metrics(config, metric_df):
    folds = list(range(config["folds"]))
    fold_options = ["All", *folds]
    st.selectbox(
        label="Folds for Class-wise Metrics", options=fold_options, key="fold_cw_metrics", index=0
    )
    eval_utils.get_class_metrics(config, metric_df, fold=st.session_state["fold_cw_metrics"])

@st.fragment()
def confusion_matrix(config, oof_df):
    folds = list(range(config["folds"]))
    fold_options = ["All", *folds]
    st.selectbox(
        label="Folds for Confusion Matrix", options=fold_options, key="fold_con_matrix", index=0
    )
    eval_utils.show_confusion_matrix(config, oof_df, fold=st.session_state["fold_con_matrix"])

@st.fragment()
def category_based_metrics(config, cleaned_df, oof_df):
    col_1, col_2, col_3 = st.columns(3)
    with col_1:
        folds = list(range(config["folds"]))
        fold_options = ["All", *folds]
        st.selectbox(
            label="Folds for Category-based Metrics", options=fold_options,
            key="fold_cb_matrix", index=0
        )
    with col_2:
        st.slider(
            "Max Unique Values",
            min_value=2, max_value=8, value=4, key="max_unique"
        )
    with col_3:
        st.toggle("Use Numeric Groups", value=False, key="bool_num")

    cat_df_dict = eval_utils.get_category_metrics(
        config, cleaned_df=copy.deepcopy(cleaned_df), oof_df=copy.deepcopy(oof_df),
        fold=st.session_state["fold_cb_matrix"], max_unique=st.session_state["max_unique"],
        numeric=st.session_state["bool_num"]
    )

    return cat_df_dict

@st.fragment()
def show_category_metrics(cat_df_dict):
    group_cols = cat_df_dict.keys()
    if len(group_cols)>0:
        st.selectbox(label="Group Column", options=group_cols, key="group_col", index=0)
        eval_utils.show_category_metrics(cat_df_dict, st.session_state["group_col"])

@st.fragment()
def training_runtime(config, metric_df):
    folds = list(range(config["folds"]))
    fold_options = ["All", *folds]
    st.selectbox(
        label="Folds for Training Runtime", options=fold_options, key="fold_runtime", index=0
    )
    eval_utils.show_runtime(metric_df, fold=st.session_state["fold_runtime"])

@st.fragment()
def reg_diagnostics(config, oof_df):
    folds = list(range(config["folds"]))
    fold_options = ["All", *folds]
    st.selectbox(
        label="Folds for Regression Diagnostics", options=fold_options, key="fold_reg_diag", index=0
    )
    eval_utils.show_reg_diagnostics(config, oof_df, fold=st.session_state["fold_reg_diag"])

@st.fragment()
def predicted_distribution(config, oof_df):
    folds = list(range(config["folds"]))
    fold_options = ["All", *folds]
    st.selectbox(
        label="Folds for Predicted Distribution", options=fold_options, key="fold_dist", index=0
    )
    if config["ml_task"]=="Classification":
        eval_utils.show_classif_predicted_distribution(
            config, oof_df, fold=st.session_state["fold_dist"]
        )
    else:
        eval_utils.show_regress_predicted_distribution(
            config, oof_df, fold=st.session_state["fold_dist"]
        )

def model_evaluation():
    exp_list = os.listdir("experiments")
    if len(exp_list)>0:
        # Exp Name for Evaluation
        st.selectbox(label="Experiment Name", options=exp_list, key="exp_name_eval", index=0)
        exp_path = os.path.join("experiments", st.session_state["exp_name_eval"])

        # Read Necessary Files
        config_filepath = os.path.join(exp_path, "config.json")
        with open(config_filepath, 'r', encoding="utf-8") as file:
            config = json.load(file)

        oof_filepath = os.path.join(exp_path, "oof_df.parquet")
        oof_df = pd.read_parquet(oof_filepath)

        metric_filepath = os.path.join(exp_path, "metric_df.parquet")
        metric_df = pd.read_parquet(metric_filepath)

        dp_path = os.path.join("datasets/cleaned_dataset", config["dp_name"])
        cleaned_df_path = os.path.join(dp_path, "cleaned_train.parquet")
        cleaned_df = pd.read_parquet(cleaned_df_path)

        # All Metrics
        st.header("All Metrics", divider="orange")
        all_metrics(config, metric_df)

        # Class-wise Metrics
        if config["ml_task"]=="Classification" and len(oof_df[config["target"]].unique()) > 2:
            st.header("Class-wise Metrics", divider="orange")
            class_wise_metrics(config, metric_df)

        # Confusion Matrix
        if config["ml_task"]=="Classification":
            st.header("Confusion Matrix", divider="orange")
            confusion_matrix(config, oof_df)

        # Category-based Matrices
        st.header("Category-based Metrics", divider="orange")
        cat_df_dict = category_based_metrics(config, cleaned_df, oof_df)
        show_category_metrics(cat_df_dict)

        # Training Runtime
        st.header("Training Runtime", divider="orange")
        training_runtime(config, metric_df)

        # Regression Diagnostics
        if config["ml_task"]=="Regression":
            st.header("Regression Diagnostics", divider="orange")
            reg_diagnostics(config, oof_df)

        # Predicted Distributions
        st.header("Predicted Distribution", divider="orange")
        predicted_distribution(config, oof_df)

    else:
        st.warning("You need to train your models to explore model evaluation")

# ==================================================================================================
# MAIN
# ==================================================================================================
# Title
st.title("Evaluation")

# Evaluation
model_evaluation()
