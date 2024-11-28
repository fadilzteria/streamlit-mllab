import os
import copy
import json
import shutil
import pandas as pd
import streamlit as st

from codes.utils import data_quality as dq, univariate_analysis as ua, test_utils, explain_utils

# ==================================================================================================
# DATASET INPUT
# ==================================================================================================
@st.fragment()
def fill_dataset_input():
    # ---------------------------------------------------
    # Select the Dataset Input Method
    data_methods = ["Select dataset from the project", "Upload a dataset file"]
    st.radio(
        label="**Please choose how you will input the test dataset**",
        options=data_methods,
        key="data_method", index=0
    )

    # ---------------------------------------------------
    # Select Dataset from the Project
    if st.session_state["data_method"]==data_methods[0]:
        raw_data_path = "datasets/raw_dataset/test_dataset"
        exts = ["csv", "pkl", "parquet", "feather"]
        all_files = os.listdir(raw_data_path)
        dataset_files = [file for file in all_files if any(ext in file for ext in exts)]
        if len(dataset_files) == 0:
            st.warning(
                "There is no dataset in the **datasets/raw_dataset/test_dataset** \
                folder. You need to put your datasets in this folder to continue \
                predict targets from your test dataset"
            )

            return

        st.selectbox(
            label="**Select a Dataset File** (*.csv, *.pkl, *.parquet, *.feather)",
            options=dataset_files,
            key="data_file", index=0
        )
        if st.session_state["data_file"]:
            st.session_state["data_filename"] = copy.deepcopy(st.session_state["data_file"])
            data_filepath = os.path.join(raw_data_path, st.session_state["data_filename"])
            raw_df = dq.load_dataframe(data_filepath)
            st.session_state["temp_raw_test_dataset"] = raw_df

    # Upload a Dataset File
    else:
        uploaded_file = st.file_uploader(label="**Upload a Dataset File**", key="data_file")
        if st.session_state["data_file"]:
            st.session_state["data_filename"] = copy.deepcopy(uploaded_file.name)
            raw_df = dq.load_dataframe(
                st.session_state["data_filename"], uploaded_file=uploaded_file
            )
            st.session_state["temp_raw_test_dataset"] = raw_df

    # Data Preprocessing Name
    dp_list = os.listdir("datasets/cleaned_dataset")
    st.selectbox(label="Data Preprocessing Name", options=dp_list, key="dp_name", index=0)

# ==================================================================================================
# DATASET PREPROCESSING
# ==================================================================================================
def preprocess_data():
    if "data_file" not in st.session_state:
        st.error("The **dataset** has not been loaded yet.")
        return

    st.session_state["raw_test_dataset"] = copy.deepcopy(st.session_state["temp_raw_test_dataset"])
    cleaned_df = copy.deepcopy(st.session_state["raw_test_dataset"])

    # Read Data Preprocessing Configuration
    dp_path = os.path.join("datasets/cleaned_dataset", st.session_state["dp_name"])
    config_filepath = os.path.join(dp_path, "df_config.json")
    with open(config_filepath, "r", encoding="utf-8") as file:
        dp_sets = json.load(file)

    # Filter Features
    cleaned_columns = dp_sets["dp_df_columns"]
    for col in ["id", "target"]:
        if dp_sets[col] not in cleaned_df.columns:
            cleaned_columns.remove(dp_sets[col])
    cleaned_df = cleaned_df[cleaned_columns]

    # Transform Value Types
    dp_num2bin_cols = dp_sets["dp_num2bin_cols"]
    if dp_sets["target"] not in cleaned_df:
        try:
            dp_num2bin_cols.pop(dp_sets["target"])
        except KeyError:
            pass
    dp_num2cat_cols = dp_sets["dp_num2cat_cols"]
    if dp_sets["target"] in dp_num2cat_cols:
        dp_num2cat_cols.remove(dp_sets["target"])
    cleaned_df, _ = dq.transform_dtypes(
        cleaned_df, split="Test",
        num2bin_cols=dp_sets["dp_num2bin_cols"], num2cat_cols=dp_num2cat_cols
    )

    # Handling Missing Values
    if "Impute" in dp_sets["dp_missed_opt"]:
        for col in dp_sets["imp_values"]:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df["imp_values"][col])

    st.session_state["cleaned_test_dataset"] = cleaned_df

    # Save Cleaned Testing Dataset
    df_filepath = os.path.join(dp_path, "cleaned_test.parquet")
    cleaned_df.to_parquet(df_filepath)

# ==================================================================================================
# DATASET
# ==================================================================================================
def submit_dataset_input():
    st.header("Dataset Input and Preprocessing", divider="orange")

    with st.container(border=True):
        fill_dataset_input()

        # ---------------------------------------------------
        # Submit
        st.button("Input Data", on_click=preprocess_data)

# ==================================================================================================
# SHOW CLEANED DATASET
# ==================================================================================================
@st.cache_data
def show_cleaned_dataset(cleaned_df):
    st.header("Cleaned Dataset", divider="orange")
    st.write(f"Shape: {cleaned_df.shape}")
    st.dataframe(cleaned_df)

# ==================================================================================================
# DATA COMPARISON
# ==================================================================================================
def data_comparison(test_df):
    if len(test_df) > 1000:
        st.header("Data Comparison", divider="orange")
        st.toggle("Using Data Comparison", value=True, key="bool_data_comparison")

        if st.session_state["bool_data_comparison"]:
            # Cleaned Train Dataframe
            dp_path = os.path.join("datasets/cleaned_dataset", st.session_state["dp_name"])
            train_df_path = os.path.join(dp_path, "cleaned_train.parquet")
            train_df = pd.read_parquet(train_df_path)

            # ---------------------------------------------------
            # Value Counts
            st.subheader("Value Counts")
            all_value_df_list = [ua.calculate_value_counts(df) for df in [train_df, test_df]]
            ua.show_value_counts(all_value_df_list, ["Train", "Test"])

            # ---------------------------------------------------
            # Box Plot
            st.subheader("Box Plot")
            ua.show_box_plot([train_df, test_df], ["Train", "Test"])

            # ---------------------------------------------------
            # Distribution
            st.subheader("Distribution")
            ua.show_kde_distribution([train_df, test_df], ["Train", "Test"])

# ==================================================================================================
# TESTING CONFIGURATION
# ==================================================================================================
@st.fragment()
def fill_testing_configuration():
    st.text_input(label="Test Name", value=None, key="test_name") # Test Name

    # Exp Name for Testing
    filtered_exp_list = []
    exp_list = os.listdir("experiments")
    for exp in exp_list:
        exp_path = os.path.join("experiments", exp)
        config_filepath = os.path.join(exp_path, "config.json")
        with open(config_filepath, 'r', encoding="utf-8") as file:
            temp_config = json.load(file)
        if st.session_state["dp_name"]==temp_config["dp_name"]:
            filtered_exp_list.append(exp)
    st.selectbox(label="Experiment Name", options=filtered_exp_list, key="exp_name", index=0)

    # Read Necessary Files
    exp_path = os.path.join("experiments", st.session_state["exp_name"])
    config_filepath = os.path.join(exp_path, "config.json")
    with open(config_filepath, 'r', encoding="utf-8") as file:
        config = json.load(file)

    # Folds
    fold_list = list(range(config["folds"]))
    st.multiselect(label="Folds", options=fold_list, default=fold_list, key="folds")

    # Models
    methods = config["methods"] # ...
    n_models = config["n_models"]
    model_names = []
    for model_name in methods:
        model_key = "_".join(model_name.lower().split(" "))
        for n in range(1, n_models[f"{model_key}_n_models"]+1):
            model_n_name = f"{model_name} {n}"
            model_names.append(model_n_name)
    st.multiselect(
        label="Model Options", options=model_names, default=model_names, key="model_names"
    )

# ==================================================================================================
# RUN TESTING
# ==================================================================================================
def run_testing():
    if st.session_state["test_name"] is None or st.session_state["test_name"]=="":
        st.error("You should put the test name properly")
        return
    if len(st.session_state["folds"])==0:
        st.error("You should pick at least one fold that you want to use.")
        return
    if len(st.session_state["model_names"])==0:
        st.error("You should pick at least one model that you want to use.")
        return

    test_df = copy.deepcopy(st.session_state["cleaned_test_dataset"])
    methods = st.session_state["model_names"]

    test_config = {
        "test_name": st.session_state["test_name"],
        "data_filename": st.session_state["data_filename"],
        "dp_name": st.session_state["dp_name"],
        "exp_name": st.session_state["exp_name"],
        "folds": st.session_state["folds"],
        "methods": methods,
    }

    config_filepath = os.path.join("experiments", test_config["exp_name"], "config.json")
    with open(config_filepath, 'r', encoding="utf-8") as file:
        train_config = json.load(file)

    # Save Test Config
    base_test_path = os.path.join("experiments", train_config["exp_name"], "tests")
    if os.path.exists(base_test_path) is False:
        os.mkdir(base_test_path)
    test_path = os.path.join(base_test_path, test_config["test_name"])
    if os.path.exists(test_path):
        shutil.rmtree(test_path)
    os.mkdir(test_path)

    test_config = dict(sorted(test_config.items()))
    config_filepath = os.path.join(test_path, "test_config.json")
    with open(config_filepath, "w", encoding="utf-8") as f:
        json.dump(test_config, f)

    # Inference
    pred_df = test_utils.inference(test_config, train_config, test_df, methods)
    st.session_state["pred_dataset"] = pred_df

    # Ensembling
    ensembled_df = test_utils.full_ensembling(test_config, train_config, test_df, pred_df)
    st.session_state["ensembled_dataset"] = ensembled_df

    # Evaluation
    if f"{train_config['target']}_actual" in ensembled_df:
        st.session_state["pred_dataset"][train_config['target']] = test_df[train_config['target']]
        metric_df = test_utils.eval_testing(test_config, train_config, ensembled_df)
        st.session_state["metric_dataset"] = metric_df
    elif "metric_dataset" in st.session_state:
        del st.session_state["metric_dataset"]

# ==================================================================================================
# MODEL TESTING
# ==================================================================================================
def model_testing():
    st.header("Testing Configuration", divider="orange")

    with st.container(border=True):
        fill_testing_configuration()

        st.button("Test the Model", on_click=run_testing)

# ==================================================================================================
# RESULTS
# ==================================================================================================
def show_testing_results():
    st.header("Results", divider="orange")

    ensembled_df = st.session_state["ensembled_dataset"]
    st.subheader("Prediction Results")
    st.dataframe(ensembled_df)

    if "metric_dataset" in st.session_state:
        full_metric_df = st.session_state["metric_dataset"]
        st.subheader("Metric Results")
        st.dataframe(full_metric_df)

# ==================================================================================================
# EXPLAINABILITY
# ==================================================================================================
@st.fragment()
def fill_explainability_configuration():
    # Read Train Config
    exp_path = os.path.join("experiments", st.session_state["exp_name"])
    config_filepath = os.path.join(exp_path, "config.json")
    with open(config_filepath, 'r', encoding="utf-8") as file:
        config = json.load(file)

    col_1, col_2 = st.columns(2)
    with col_1:
        # Folds
        fold_options = st.session_state["folds"]
        st.selectbox(
            label="Folds for Model Explainability", options=fold_options, key="fold_modelx", index=0
        )
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

    if len(tree_model_names) == 0:
        st.warning(
            "You need to have ensemble tree models such as XGBoost, LightGBM, \
            and CatBoost from your experiment to explore model explainability"
        )

def run_explainability():
    if st.session_state["tree_model"] is not None:
        explain_config = {
            "exp_name": st.session_state["exp_name"],
            "tree_model": st.session_state["tree_model"],
            "fold_modelx": st.session_state["fold_modelx"],
        }

        bool_label = any("actual" in col for col in st.session_state["ensembled_dataset"].columns)
        if bool_label:
            results = explain_utils.explain_model(explain_config, split="Valid")
        else:
            results = explain_utils.explain_model(explain_config, split="Test")

        st.session_state["ex_model"], st.session_state["explainer"] = results[0:2]
        st.session_state["explanation_test"], st.session_state["shap_values"] = results[2:4]
        st.session_state["X_data"], st.session_state["unique_targets"] = results[4:]
    else:
        st.error(
            "You need to have ensemble tree models such as XGBoost, LightGBM, \
            and CatBoost from your experiment to explore model explainability"
        )

def model_explainability():
    st.subheader("Configuration")
    with st.container(border=True):
        fill_explainability_configuration()
        st.button("Explain the Model", on_click=run_explainability)

@st.fragment()
def local_explainability():
    max_idx = len(st.session_state["X_data"]) - 1
    st.number_input("Index Dataframe", min_value=0, max_value=max_idx, value=0, key="idx")
    explain_utils.show_local_explainability(
        st.session_state["explainer"], st.session_state["explanation_test"],
        st.session_state["shap_values"], st.session_state["X_data"],
        st.session_state["unique_targets"], idx=st.session_state["idx"]
    )

def show_explainability():
    # Read Train Config
    exp_path = os.path.join("experiments", st.session_state["exp_name"])
    config_filepath = os.path.join(exp_path, "config.json")
    with open(config_filepath, 'r', encoding="utf-8") as file:
        config = json.load(file)

    # Read Pred Dataset
    pred_df = st.session_state["pred_dataset"]

    # Easy-Difficult Samples
    bool_label = any("actual" in col for col in st.session_state["ensembled_dataset"].columns)
    if bool_label:
        st.subheader("Easy-Difficult Samples")
        spec_model_name = f"{st.session_state['tree_model']}_{st.session_state['fold_modelx']}"
        explain_utils.extract_easy_difficult_samples(
            config, pred_df, spec_model_name, st.session_state["fold_modelx"], split="Valid"
        )

    # Local
    # ------------------------------------------------------
    st.header("Local", divider="orange")
    local_explainability()

# ==================================================================================================
# PREDICTION
# ==================================================================================================
def prediction():
    exp_list = os.listdir("experiments")
    if len(exp_list)>0:
        # Dataset Input
        submit_dataset_input()

        # Show Cleaned Dataset
        if "cleaned_test_dataset" in st.session_state:
            show_cleaned_dataset(st.session_state["cleaned_test_dataset"])
            data_comparison(st.session_state["cleaned_test_dataset"])

        # Testing Configuration
        if "cleaned_test_dataset" in st.session_state:
            model_testing()

        if "pred_dataset" in st.session_state:
            # Show Results
            show_testing_results()

            # Local Explainability
            st.header("Local Explainability", divider="orange")
            model_explainability()
            if "explanation_test" in st.session_state:
                show_explainability()

    else:
        st.warning("You need to train your models to predict targets from your test dataset")

# ==================================================================================================
# MAIN
# ==================================================================================================
# Title
st.title("Prediction")

# Prediction
prediction()
