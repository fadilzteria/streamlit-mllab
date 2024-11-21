import os
import copy
import json
import pandas as pd
import streamlit as st
import shutil

from codes.utils import data_quality as dq, univariate_analysis as ua, test_utils

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
    if(st.session_state["data_method"]==data_methods[0]):
        raw_data_path = "datasets/raw_dataset/test_dataset"
        exts = ["csv", "pkl", "parquet", "feather"]
        dataset_files = [file for file in os.listdir(raw_data_path) if any(ext in file for ext in exts)]
        st.selectbox(
            label="**Select a Dataset File** (*.csv, *.pkl, *.parquet, *.feather)",
            options=dataset_files,
            key="data_file", index=0
        )
        if(st.session_state["data_file"]):
            st.session_state["data_filename"] = copy.deepcopy(st.session_state["data_file"])
            data_filepath = os.path.join(raw_data_path, st.session_state["data_filename"])
            raw_df = dq.load_dataframe(data_filepath)
            st.session_state["temp_raw_test_dataset"] = raw_df

    # Upload a Dataset File
    else:
        uploaded_file = st.file_uploader(label="**Upload a Dataset File**", key="data_file")
        if(st.session_state["data_file"]):
            st.session_state["data_filename"] = copy.deepcopy(uploaded_file.name)
            raw_df = dq.load_dataframe(st.session_state["data_filename"], uploaded_file=uploaded_file)
            st.session_state["temp_raw_test_dataset"] = raw_df

    # Data Preprocessing Name
    dp_list = os.listdir("datasets/cleaned_dataset")
    st.selectbox(label="Data Preprocessing Name", options=dp_list, key="dp_name", index=0)

# ==================================================================================================
# DATASET PREPROCESSING
# ==================================================================================================
def preprocess_data():
    st.session_state["raw_test_dataset"] = copy.deepcopy(st.session_state["temp_raw_test_dataset"])
    cleaned_df = copy.deepcopy(st.session_state["raw_test_dataset"])

    # Read Data Preprocessing Configuration
    dp_path = os.path.join("datasets/cleaned_dataset", st.session_state["dp_name"])
    config_filepath = os.path.join(dp_path, "df_config.json")
    with open(config_filepath, "r") as file:
        dp_sets = json.load(file)

    # Filter Features
    cleaned_columns = dp_sets["dp_df_columns"]
    for col in ["id", "target"]:
        if(dp_sets[col] not in cleaned_df.columns):
            cleaned_columns.remove(dp_sets[col])
    cleaned_df = cleaned_df[cleaned_columns]

    # Transform Value Types
    dp_num2bin_cols = dp_sets["dp_num2bin_cols"]
    if(dp_sets["target"] not in cleaned_df):
        try:
            dp_num2bin_cols.pop(dp_sets["target"])
        except:
            pass
    dp_num2cat_cols = dp_sets["dp_num2cat_cols"]
    if(dp_sets["target"] in dp_num2cat_cols):
        dp_num2cat_cols.remove(dp_sets["target"])
    cleaned_df, _ = dq.transform_dtypes(
        cleaned_df, split="Test", 
        num2bin_cols=dp_sets["dp_num2bin_cols"], num2cat_cols=dp_num2cat_cols
    )

    # Handling Missing Values
    if("Impute" in dp_sets["dp_missed_opt"]):
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
def show_cleaned_dataset(cleaned_df):
    st.header("Cleaned Dataset", divider="orange")
    st.write(f"Shape: ", cleaned_df.shape)
    st.dataframe(cleaned_df)   

# ==================================================================================================
# DATA COMPARISON
# ==================================================================================================
def data_comparison(test_df):
    if(len(test_df) > 1000):
        st.header("Data Comparison", divider="orange")

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
    st.text_input(label="Test Name", value="Test 1", key="test_name") # Test Name

    # Exp Name for Testing
    exp_list = os.listdir("experiments")
    st.selectbox(label="Experiment Name", options=exp_list, key="exp_name", index=0) 
    exp_path = os.path.join("experiments", st.session_state["exp_name"])

    # Read Necessary Files
    config_filepath = os.path.join(exp_path, "config.json")
    with open(config_filepath, 'r') as file:
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
    st.multiselect(label="Model Options", options=model_names, default=model_names, key="model_names")

# ==================================================================================================
# RUN TESTING
# ==================================================================================================
def run_testing():  
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
    with open(config_filepath, 'r') as file:
        train_config = json.load(file)

    # Save Test Config
    base_test_path = os.path.join("experiments", train_config["exp_name"], "tests")
    if(os.path.exists(base_test_path) == False):
        os.mkdir(base_test_path)
    test_path = os.path.join(base_test_path, test_config["test_name"])
    if(os.path.exists(test_path)):
        shutil.rmtree(test_path)
    os.mkdir(test_path)

    test_config = dict(sorted(test_config.items()))
    config_filepath = os.path.join(test_path, "test_config.json")
    with open(config_filepath, "w") as f:
        json.dump(test_config, f)

    # Inference
    pred_df = test_utils.inference(test_config, train_config, test_df, methods)
    st.session_state["pred_dataset"] = pred_df

    # Ensembling
    ensembled_df = test_utils.full_ensembling(test_config, train_config, test_df, pred_df)
    st.session_state["ensembled_dataset"] = ensembled_df

    # Evaluation
    if(f"{train_config['target']}_actual" in ensembled_df):
        metric_df = test_utils.eval_testing(test_config, train_config, ensembled_df)
        st.session_state["metric_dataset"] = metric_df
    elif("metric_dataset" in st.session_state):
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

    if("metric_dataset" in st.session_state):
        full_metric_df = st.session_state["metric_dataset"]
        st.subheader("Metric Results")
        st.dataframe(full_metric_df)

# ==================================================================================================
# PREDICTION
# ==================================================================================================
def prediction():
    exp_list = os.listdir("experiments")
    if(len(exp_list)>0):
        # Dataset Input
        submit_dataset_input()

        # Show Cleaned Dataset
        if("cleaned_test_dataset" in st.session_state): 
            show_cleaned_dataset(st.session_state["cleaned_test_dataset"])
            data_comparison(st.session_state["cleaned_test_dataset"])

        # Testing Configuration
        if("cleaned_test_dataset" in st.session_state):
            model_testing()

        # Show Results
        if("pred_dataset" in st.session_state):
            show_testing_results()
            
    else:
        st.warning("You need to train your models to predict targets from your test dataset")

# ==================================================================================================
# MAIN
# ==================================================================================================
# Title
st.title("Prediction")

# Prediction
prediction()