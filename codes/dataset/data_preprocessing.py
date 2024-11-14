import os
import copy
import shutil
import json
import pandas as pd
import streamlit as st

from codes.utils import data_quality as dq

# ==================================================================================================
# DATASET INPUT
# ==================================================================================================
@st.fragment()
def fill_dataset_input():
    # ---------------------------------------------------
    # Select the Dataset Input Method
    data_methods = ["Select dataset from the project", "Upload a dataset file"]
    st.radio(
        label="**Please choose how you will input the raw dataset**",
        options=data_methods,
        key="data_method", index=0
    )

    # ---------------------------------------------------
    # Select Dataset from the Project
    if(st.session_state["data_method"]==data_methods[0]):
        raw_data_path = "datasets/raw_dataset"
        dataset_files = os.listdir(raw_data_path)
        st.selectbox(
            label="**Select a Dataset File**",
            options=dataset_files,
            key="data_file", index=0
        )
        if(st.session_state["data_file"]):
            data_filepath = os.path.join(raw_data_path, st.session_state["data_file"])
            raw_df = pd.read_csv(data_filepath)
            st.session_state["temp_raw_train_dataset"] = raw_df

    # Upload a Dataset File
    else:
        uploaded_file = st.file_uploader(label="**Upload a Dataset File**", key="data_file")
        if(st.session_state["data_file"]):
            raw_df = pd.read_csv(uploaded_file)
            st.session_state["temp_raw_train_dataset"] = raw_df

    # ---------------------------------------------------
    # Feature Names
    if("temp_raw_train_dataset" in st.session_state):
        df_columns = st.session_state["temp_raw_train_dataset"].columns.tolist()
        st.selectbox(label="ID Column", options=df_columns, key="_id", index=None)
        st.selectbox(label="Target Column", options=df_columns, key="_target", index=None)

def check_dataset_input():
    if(st.session_state["_target"]):
        st.session_state["id"] = st.session_state["_id"]
        st.session_state["target"] = st.session_state["_target"]
        st.session_state["raw_train_dataset"] = copy.deepcopy(st.session_state["temp_raw_train_dataset"])
        if(st.session_state["id"]==None):
            st.session_state["raw_train_dataset"] = st.session_state["raw_train_dataset"].reset_index(names="id")
            st.session_state["id"] = "id"

def submit_dataset_input():
    st.header("Dataset Input", divider="orange")
    
    with st.container(border=True):
        fill_dataset_input()

        # ---------------------------------------------------
        # Submit
        st.button("Input Data", on_click=check_dataset_input)

# ==================================================================================================
# SHOW RAW DATASET
# ==================================================================================================
@st.cache_data()
def show_raw_dataset(raw_df):
    st.header("Raw Dataset", divider="orange")

    # ---------------------------------------------------
    # Dataframe
    st.subheader("Dataframe")
    st.write(f"Shape: ", raw_df.shape)
    st.dataframe(raw_df) # Show Dataframe

    # ---------------------------------------------------
    # Data Quality
    st.subheader("Data Quality")

    # Unique, Missing, Infinite, Zero, and Negative Values
    for value in ["Unique", "Missing", "Infinite", "Zero", "Negative"]:
        if(value=="Unique"):
            drop_features = [st.session_state["id"], st.session_state["target"]]
            value_df = dq.check_other_values(raw_df, value, drop_features=drop_features, max_unique=len(raw_df))
        else:
            value_df = dq.check_other_values(raw_df, value)
        dq.show_other_values([value_df], ["Raw Dataset"])    

    # Duplicated Rows
    raw_df = dq.check_duplicated_rows(raw_df, target_exist=st.session_state["target"], spec_features=[st.session_state["id"]])

# ==================================================================================================
# DATA PREPROCESSING
# ==================================================================================================
def preprocess_data():
    cleaned_df = copy.deepcopy(st.session_state["raw_train_dataset"])

    # Drop Constant and Full Unique Features
    if(st.session_state["dp_bool_unique"]):
        persisted_features = [st.session_state["id"], st.session_state["target"]]
        cleaned_df = dq.drop_unique_features(cleaned_df, persisted_features)
    st.session_state["dp_df_columns"] = cleaned_df.columns.tolist()

    # Drop Duplicated Rows
    if(st.session_state["dp_bool_dup"]):
        cleaned_df = dq.check_duplicated_rows(
            cleaned_df, drop_duplicated_rows=True, 
            target_exist=st.session_state["target"], spec_features=[st.session_state["id"]]
        )    

    # Transform Value Types
    cleaned_df, st.session_state["dp_num2bin_cols"] = dq.transform_dtypes(
        cleaned_df, num2cat_cols=st.session_state["dp_num2cat_cols"]
    )

    # Handling Missing Values
    if("Drop" in st.session_state["dp_missed_opt"]):
        if(st.session_state["dp_missed_opt"]=="Drop Missing Features"):
            cleaned_df = cleaned_df.dropna(axis=1)
        else:
            cleaned_df = cleaned_df.dropna(axis=0)
    elif("Impute" in st.session_state["dp_missed_opt"]):
        impute = st.session_state["dp_missed_opt"].split("_")[-1]
        cleaned_df, imp_values = dq.impute_missing_values(cleaned_df, impute)
        st.session_state["dp_imp_values"] = imp_values

    # Use Sample Data
    if(st.session_state["dp_bool_sample"]):
        cleaned_df = dq.sampling_data(
            cleaned_df, st.session_state["dp_sample_size"], 
            id=st.session_state["id"], group=st.session_state["target"])

    st.session_state["cleaned_train_dataset"] = cleaned_df

    # Save Data Preprocessing Config and Cleaned Dataset
    dp_sets = {key: st.session_state[key] for key in st.session_state.keys() if "dp" in key}
    dp_sets["id"] = st.session_state["id"]
    dp_sets["target"] = st.session_state["target"]

    dp_path = os.path.join("datasets/cleaned_dataset", dp_sets["dp_name"])
    if(os.path.exists(dp_path)):
        shutil.rmtree(dp_path)
    os.mkdir(dp_path)

    df_filepath = os.path.join(dp_path, "cleaned_train.csv")
    cleaned_df.to_csv(df_filepath, index=False)
    
    config_filepath = os.path.join(dp_path, "df_config.json")
    with open(config_filepath, "w") as f:
        json.dump(dp_sets, f)

def data_preprocessing(raw_df):
    st.header("Data Preprocessing", divider="orange")

    with st.form("data_preprocessing"):
        st.text_input(label="Data Preprocessing Name", value="Baseline", key="dp_name") # Data Preprocessing Name
        st.toggle("Drop Constant and Full Unique Features", value=True, key="dp_bool_unique")
        st.toggle("Drop Duplicated Rows", value=True, key="dp_bool_dup")

        num_columns = []
        for col in raw_df.columns:
            if(raw_df[col].dtypes in ["int", "float"] and raw_df[col].nunique() > 2):
                num_columns.append(col)
        for col in ["id", "target"]:
            if(col in num_columns):
                num_columns.remove(col)
        st.multiselect(label="Numerical to Categorical Columns", options=num_columns, default=[], key="dp_num2cat_cols")

        missed_options = [
            "Ignore",
            "Drop Missing Features",
            "Drop Missing Rows",
            "Impute by Mean/Mode",
            "Impute by Median/Mode"
        ]
        st.selectbox(label="Handling Missing Values", options=missed_options, key="dp_missed_opt", index=0)
        col_1, col_2 = st.columns(2)
        with col_1:
            st.toggle("Use Sample Data", value=False, key="dp_bool_sample")
        with col_2:
            st.number_input(
                "Sample Size", 
                min_value=1, max_value=len(st.session_state["raw_train_dataset"]), 
                value=None, key="dp_sample_size"
            )

        st.form_submit_button("Process Data", on_click=preprocess_data)

# ==================================================================================================
# SHOW CLEANED DATASET
# ==================================================================================================
# @st.cache_data()
def show_cleaned_dataset(cleaned_df):
    st.header("Cleaned Dataset", divider="orange")
    st.write(f"Shape: ", cleaned_df.shape)
    st.dataframe(cleaned_df)    

# ==================================================================================================
# MAIN
# ==================================================================================================
# Title
st.title("Dataset") 

# Dataset Input
submit_dataset_input()

if("raw_train_dataset" in st.session_state): 
    show_raw_dataset(st.session_state["raw_train_dataset"]) # Show Raw Dataset
    data_preprocessing(st.session_state["raw_train_dataset"]) # Data Preprocessing

# Show Cleaned Ddataset
if("cleaned_train_dataset" in st.session_state): 
    show_cleaned_dataset(st.session_state["cleaned_train_dataset"])