import os
import copy
import pandas as pd
import streamlit as st

from codes.utils import bivariate_analysis as ba

# ==================================================================================================
# BIVARIATE ANALYSIS
# ==================================================================================================
@st.fragment()
def correlation_matrix(all_matrices):
    corr_types = ["Pearson", "Spearman", "Kendall"]
    st.selectbox(label="Corr Type for Correlation_matrix", options=corr_types, key="corr_type_cm", index=0)
    ba.show_correlation_heatmap(all_matrices, st.session_state["corr_type_cm"])

@st.fragment()
def nums_correlation(cleaned_df, corr_matrix_df, new_columns):
    with st.container(border=True):
        col_1, col_2 = st.columns(2)
        with col_1:
            corr_types = ["Pearson", "Spearman", "Kendall"]
            st.selectbox(label="Corr Type for Two Nums Correlation", options=corr_types, key="corr_type_nums", index=0)
        with col_2:
            st.slider(
                "Max Correlation for each Correlation Strength",
                min_value=1, max_value=12, value=8, key="max_corr"
            )

        col_1, col_2 = st.columns(2)
        with col_1:
            st.slider(
                "Min Strong Positive Correlation",
                min_value=0.20, max_value=1.00, value=0.60, step=0.01, key="pos_value"
            )
        with col_2:
            st.slider(
                "Min Strong Negative Correlation",
                min_value=-1.00, max_value=-0.20, value=-0.60, step=0.01, key="neg_value"
            )
        
        col_1, col_2 = st.columns(2)
        with col_1:
            st.slider(
                "Min Weak-No Correlation",
                min_value=-0.20, max_value=0.00, value=-0.10, step=0.01, key="min_zero_value"
            )
        with col_2:
            st.slider(
                "Max Weak-No Correlation",
                min_value=0.00, max_value=0.20, value=0.10, step=0.01, key="max_zero_value"
            )

    strong_weak_dfs, titles = ba.extract_strong_weak_corr_matrix(
        corr_matrix_df, new_columns, 
        st.session_state["corr_type_nums"], max_corr=st.session_state["max_corr"],
        pos_value=st.session_state["pos_value"], neg_value=st.session_state["neg_value"], 
        min_zero_value=st.session_state["min_zero_value"], max_zero_value=st.session_state["max_zero_value"]
    )
    ba.show_correlation_scatter_plot(cleaned_df, strong_weak_dfs, titles)

@st.fragment()
def num_cat_correlation(cleaned_df):
    with st.container(border=True):
        col_1, col_2, col_3 = st.columns(3)
        with col_1:
            st.slider(
                "Sample Size",
                min_value=1, max_value=500, value=50, key="sample_size"
            )
        with col_2:
            st.slider(
                "Max Unique Values for Cat Columns",
                min_value=2, max_value=10, value=5, key="max_unique"
            )
        with col_3:
            st.slider(
                "Max Number Plots",
                min_value=2, max_value=20, value=6, key="num_plots"
            )
        
        col_1, col_2, col_3 = st.columns(3)
        with col_1:
            st.slider(
                "Min Samples from Sample Mean",
                min_value=20, max_value=1000, value=20, key="min_sample_mean"
            )
        with col_2:
            st.toggle("Using Max Sample Mean", value=False, key="bool_max_sample_mean")
        
        if(st.session_state["bool_max_sample_mean"]):
            with col_3:
                st.slider(
                    "Max Samples from Sample Mean",
                    min_value=st.session_state["min_sample_mean"], max_value=10000, 
                    value=2000, key="max_sample_mean"
                )
        else:
            st.session_state["max_sample_mean"] = None    

    anova_results, all_group_means = ba.test_numerical_categorical(
        cleaned_df, sample_size=st.session_state["sample_size"], max_unique=st.session_state["max_unique"],
        min_sample_mean=st.session_state["min_sample_mean"], max_sample_mean=st.session_state["max_sample_mean"]
    )
    ba.show_correlation_num_cat(
        anova_results, all_group_means, 
        num_groups=st.session_state["max_unique"], num_plots=st.session_state["num_plots"]
    )

@st.fragment()
def cats_correlation(cleaned_df):
    col_1, col_2, col_3 = st.columns(3)
    with col_1:
        st.slider(
            "Max Unique Values for Cat Columns",
            min_value=2, max_value=20, value=10, key="max_cat_unique"
        )
    with col_2:
        st.slider(
            "Num Plots from Two Binary",
            min_value=1, max_value=30, value=6, key="num_two_binary"
        )
    with col_3:
        st.slider(
            "Num Plots from At Least One Nominal",
            min_value=1, max_value=20, value=5, key="num_one_nominal"
        )
    
    chi_results = ba.test_two_categorical(cleaned_df, max_unique=st.session_state["max_cat_unique"])
    ba.show_correlation_two_binary(
        cleaned_df, chi_results, num_two_binary=st.session_state["num_two_binary"]
    )
    ba.show_correlation_least_one_nominal(
        cleaned_df, chi_results, num_one_nominal=st.session_state["num_one_nominal"]
    )

def bivariate_analysis():
    # ---------------------------------------------------
    # Dataframe
    dp_list = os.listdir("datasets/cleaned_dataset")
    st.selectbox(label="Data Preprocessing Name", options=dp_list, key="dp_name", index=0)
    dp_path = os.path.join("datasets/cleaned_dataset", st.session_state["dp_name"])
    cleaned_df_path = os.path.join(dp_path, "cleaned_train.parquet")
    cleaned_df = pd.read_parquet(cleaned_df_path)

    # ---------------------------------------------------
    # Correlation Matrix
    st.header("Correlation Matrix", divider="orange")
    corr_matrix_df, all_matrices, new_columns = ba.extract_correlation_matrix(cleaned_df)
    correlation_matrix(all_matrices)

    # ---------------------------------------------------
    # Numerical x Numerical
    st.header("Numerical x Numerical", divider="orange")
    nums_correlation(cleaned_df, corr_matrix_df, new_columns)

    # ---------------------------------------------------
    # Numerical x Categorical
    st.header("Numerical x Categorical", divider="orange")
    num_cat_correlation(cleaned_df)

    # ---------------------------------------------------
    # Categorical x Categorical
    st.header("Categorical x Categorical", divider="orange")
    cats_correlation(cleaned_df)

# ==================================================================================================
# MAIN
# ==================================================================================================
# Title
st.title("EDA - Bivariate Analysis") 

# Bivariate Analysis
dp_list = os.listdir("datasets/cleaned_dataset")
if(len(dp_list)>0):
    bivariate_analysis()
else:
    st.warning("You need to preprocess your dataset before exploring more data insights")