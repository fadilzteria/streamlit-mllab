import os
import copy
import pandas as pd
import streamlit as st

from codes.utils import train_utils

# ==================================================================================================
# TRAINING CONFIGURATION
# ==================================================================================================
@st.fragment()
def fill_training_configuration():
    # ---------------------------------------------------
    # Experiment
    st.subheader("Experiment")
    st.text_input(label="Experiment Name", value=None, key="exp_name") # Experiment Name

    st.selectbox(label="Data Preprocessing Name", options=dp_list, key="dp_name", index=0)
    dp_path = os.path.join("datasets/cleaned_dataset", st.session_state["dp_name"])
    cleaned_df_path = os.path.join(dp_path, "cleaned_train.parquet")
    cleaned_df = pd.read_parquet(cleaned_df_path)
    st.session_state["cleaned_dataset"] = cleaned_df

    st.selectbox( # ML Task
        label="ML Task (adjust with your target dataset)", options=["Classification", "Regression"],
        key="ml_task", index=0
    )
    st.number_input( # Stratified K-Fold
        label="Number Folds for Cross Validation Stratified K-Fold", value=5,
        min_value=2, max_value=15, key="folds"
    )

    # ---------------------------------------------------
    # Feature Engineering
    st.subheader("Feature Engineering")

    # cap outliers multibox; each feature with cap value
    # ...

    # Feature Creation
    col_1, col_2, col_3 = st.columns(3)
    with col_1:
        st.toggle("Feature Creation", value=False, key="fe_feat_creation")
    if st.session_state["fe_feat_creation"]:
        with col_2:
            st.toggle("Math Features", value=False, key="fe_math_creation")
        with col_3:
            st.toggle("Cat Group Features", value=False, key="fe_cat_group_creation")

    # Feature Extraction
    col_1, col_2 = st.columns(2)
    with col_1:
        st.toggle("Feature Extraction", value=False, key="fe_feat_extraction")
    if st.session_state["fe_feat_extraction"]:
        with col_2:
            st.toggle("Datetime Features", value=False, key="fe_datetime")

    # Drop Many Unique Categorical Columns
    col_1, col_2 = st.columns(2)
    with col_1:
        st.toggle(
            "Drop Many Unique Categorical Columns", value=False, key="fe_drop_many_cat_unique"
        )
    if st.session_state["fe_drop_many_cat_unique"]:
        with col_2:
            st.number_input(label="Maximal Unique Values", value=10, key="fe_max_cat_unique")

    # Categorical Encoding
    col_1, col_2 = st.columns(2)
    with col_1:
        st.toggle("Categorical Encoding", value=True, key="fe_cat_encoding")
    if st.session_state["fe_cat_encoding"]:
        with col_2:
            st.selectbox(
                label="Encoding Types", options=["One-hot", "Categorical"],
                key="fe_cat_type", index=0
            )

    # Feature Scaling
    st.toggle("Feature Scaling", value=False, key="fe_scaling")

    # ---------------------------------------------------
    # Model
    st.subheader("Model")

    # Model Names
    classif_model_options = [
        "Logistic Regression", "Linear Discriminant Analysis", 
        "Bernoulli Bayes", "Gaussian Bayes",
        "KNN", "SVC", "Linear SVC", 
        "Decision Tree", "Extra Trees", "Random Forest", "AdaBoost", "Gradient Boosting",
        "XGBoost", "LightGBM", "CatBoost"
    ]
    regress_model_options = [
        "Linear Regression", "Ridge Regression", "Lasso", "Elastic Net",
        "KNN", "SVR", "Linear SVR", 
        "Decision Tree", "Extra Trees", "Random Forest", "AdaBoost", "Gradient Boosting",
        "XGBoost", "LightGBM", "CatBoost"
    ]

    if st.session_state["ml_task"]=="Classification": # Classification
        model_options = copy.deepcopy(classif_model_options)
    else: # Regression
        model_options = copy.deepcopy(regress_model_options)
    st.multiselect(label="Model Options", options=model_options, key="model_names")

    # Number of Models and Parameters
    for model_name in st.session_state["model_names"]:
        with st.expander(model_name):
            model_key = "_".join(model_name.lower().split(" "))
            st.number_input(
                label="Number of Models", min_value=1, max_value=10,
                value=1, key=f"{model_key}_n_models"
            )

            for n in range(1, st.session_state[f"{model_key}_n_models"]+1):
                model_n_name = f"{model_name} {n}"
                model_n_key = "_".join(model_n_name.lower().split(" "))
                st.write(model_n_name)

                if model_name=="KNN":
                    st.number_input(
                        label="Number of Neighbors", value=5, key=f"{model_n_key}_n_neighbors"
                    )

                ensemble_tree_list = [
                    "Decision Tree", "Extra Trees", "Random Forest", "Gradient Boosting"
                ]
                if model_name in ensemble_tree_list:
                    st.number_input(
                        label="Max Depth", value=5, key=f"{model_n_key}_params_max_depth"
                    )

    # Metrics
    classif_metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "Avg Precision"]
    regress_metrics = ["MSE", "RMSE", "MAE", "MedAE", "R2", "Adj R2"]

    col_1, col_2 = st.columns(2)
    with col_1: # Metric Selection
        if st.session_state["ml_task"]=="Classification": # Classification
            st.multiselect(label="Metric Options", options=classif_metrics, key="metric_names")
        else: # Regression
            st.multiselect(label="Metric Options", options=regress_metrics, key="metric_names")
    with col_2: # Best Metrics
        st.selectbox(
            label="Best Metric", options=st.session_state["metric_names"],
            key="best_metric", index=0
        )

# ==================================================================================================
# RUN TRAINING
# ==================================================================================================
def run_training():
    if st.session_state["exp_name"] is None or st.session_state["exp_name"]=="":
        st.error("You should put the experiment name properly")
        return
    if len(st.session_state["model_names"])==0:
        st.error("You should pick at least one model that you want to train.")
        return
    if len(st.session_state["metric_names"])==0:
        st.error("You should pick at least one metric that you want to evaluate.")
        return
    if st.session_state["fe_cat_encoding"] and st.session_state["fe_cat_type"]=="Categorical":
        adv_ensemble_trees = ["XGBoost", "LightGBM", "CatBoost"]
        for model_name in st.session_state["model_names"]:
            if model_name not in adv_ensemble_trees:
                st.error(
                    "Categorical from categorical encoding types can only be applied \
                    for advanced ensemble tree models such as XGBoost, LightGBM, and CatBoost"
                )
                return

    train_df = copy.deepcopy(st.session_state["cleaned_dataset"])
    fe_sets = {key: st.session_state[key] for key in st.session_state if "fe" in key}

    methods = st.session_state["model_names"]
    n_models = {key: st.session_state[key] for key in st.session_state if "n_models" in key}
    n_models = dict(sorted(n_models.items()))
    params = {key: st.session_state[key] for key in st.session_state if "params" in key}
    params = dict(sorted(params.items()))

    metrics = st.session_state["metric_names"]
    if st.session_state["best_metric"] is None:
        st.session_state["best_metric"] = st.session_state["metric_names"][0]
    max_metrics = [
        "Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "Avg Precision", "R2", "Adj R2"
    ]
    if st.session_state["best_metric"] in max_metrics:
        st.session_state["best_value"] = "Maximize"
    else:
        st.session_state["best_value"] = "Minimize"

    config = {
        "exp_name": st.session_state["exp_name"], "dp_name": st.session_state["dp_name"],
        "ml_task" : st.session_state["ml_task"], "folds": st.session_state["folds"],
        "methods": methods, "n_models": n_models, "params": params, "metrics": metrics,
        "best_metric": st.session_state["best_metric"],
        "best_value": st.session_state["best_value"],
    }

    train_utils.training_and_validation(config, train_df, fe_sets)

# ==================================================================================================
# MODEL TRAINING
# ==================================================================================================
def model_training():
    st.header("Training Configuration", divider="orange")

    with st.container(border=True):
        fill_training_configuration()

        st.button("Train the Model", on_click=run_training)

# ==================================================================================================
# RESULTS
# ==================================================================================================
def show_training_dataframe():
    exp_path = os.path.join("experiments", st.session_state["exp_name"])
    oof_df_filepath = os.path.join(exp_path, "oof_df.parquet")
    if os.path.exists(oof_df_filepath):
        st.header("Training Results", divider="orange")

        oof_df = pd.read_parquet(oof_df_filepath)
        st.subheader("Out of Fold Predictions")
        st.dataframe(oof_df)

        metric_df_filepath = os.path.join(exp_path, "metric_df.parquet")
        metric_df = pd.read_parquet(metric_df_filepath)
        st.subheader("Metric Results")
        st.dataframe(metric_df)

# ==================================================================================================
# MAIN
# ==================================================================================================
# Title
st.title("Model Training")

# Training
dp_list = os.listdir("datasets/cleaned_dataset")
if len(dp_list)>0:
    model_training()

    exp_list = os.listdir("experiments")
    if st.session_state["exp_name"] in exp_list:
        show_training_dataframe()

else:
    st.warning("You need to preprocess your dataset before training models")
