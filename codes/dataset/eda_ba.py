import os
import copy
import pandas as pd
import streamlit as st

from codes.utils import bivariate_analysis as ba

# ==================================================================================================
# BIVARIATE ANALYSIS
# ==================================================================================================
# @st.cache_data()
def bivariate_analysis():
    # ---------------------------------------------------
    # Dataframe
    dp_list = os.listdir("datasets/cleaned_dataset")
    st.selectbox(label="Data Preprocessing Name", options=dp_list, key="dp_name", index=0)
    dp_path = os.path.join("datasets/cleaned_dataset", st.session_state["dp_name"])
    cleaned_df_path = os.path.join(dp_path, "cleaned_train.csv")
    cleaned_df = pd.read_csv(cleaned_df_path)

    # ---------------------------------------------------
    # Correlation Matrix
    st.header("Correlation Matrix", divider="orange")
    corr_matrix_df, all_matrices, new_columns = ba.extract_correlation_matrix(cleaned_df)
    corr_type = "Pearson"
    ba.show_correlation_heatmap(all_matrices, corr_type)

    # # ---------------------------------------------------
    # # Numerical x Numerical
    # st.header("Numerical x Numerical", divider="orange")
    # corr_type = "Pearson"
    # max_corr = 8
    # pos_value = 0.50
    # neg_value = -0.10
    # min_zero_value = -0.05
    # max_zero_value = 0.05
    # strong_weak_dfs, titles = ba.extract_strong_weak_corr_matrix(
    #     corr_matrix_df, new_columns, corr_type, max_corr=max_corr,
    #     pos_value=pos_value, neg_value=neg_value, min_zero_value=min_zero_value, max_zero_value=max_zero_value
    # )
    # ba.show_correlation_scatter_plot(cleaned_df, strong_weak_dfs, titles)

    # # ---------------------------------------------------
    # # Numerical x Categorical
    # st.header("Numerical x Categorical", divider="orange")
    
    # sample_size = 50
    # min_sample_mean = 50
    # max_sample_mean = None
    # max_unique = 10

    # anova_results, all_group_means = ba.test_numerical_categorical(cleaned_df, sample_size=sample_size, min_sample_mean=min_sample_mean, max_sample_mean=max_sample_mean, max_unique=max_unique)
    # ba.show_correlation_num_cat(anova_results, all_group_means, num_groups=10, num_plots=10)

    # # ---------------------------------------------------
    # # Categorical x Categorical
    # st.header("Categorical x Categorical", divider="orange")

    # chi_results = ba.test_two_categorical(cleaned_df)
    # ba.show_correlation_two_binary(cleaned_df, chi_results)
    # ba.show_correlation_least_one_nominal(cleaned_df, chi_results)

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