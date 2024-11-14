import os
import pandas as pd
import streamlit as st
from codes.utils import univariate_analysis as ua

# ==================================================================================================
# UNIVARIATE ANALYSIS
# ==================================================================================================
# @st.cache_data()
def univariate_analysis():
    # ---------------------------------------------------
    # Dataframe
    dp_list = os.listdir("datasets/cleaned_dataset")
    st.selectbox(label="Data Preprocessing Name", options=dp_list, key="dp_name", index=0)
    dp_path = os.path.join("datasets/cleaned_dataset", st.session_state["dp_name"])
    cleaned_df_path = os.path.join(dp_path, "cleaned_train.parquet")
    cleaned_df = pd.read_parquet(cleaned_df_path)

    # ---------------------------------------------------
    # Value Counts
    st.header("Value Counts", divider="orange")
    all_value_df_list = [ua.calculate_value_counts(df) for df in [cleaned_df]]
    ua.show_value_counts(all_value_df_list, ["Cleaned Dataset"])

    # ---------------------------------------------------
    # Statistics
    st.header("Statistics", divider="orange")
    desc_stats_df, quan_stats_df = ua.extract_statistics(cleaned_df)
    st.subheader("Descriptive Statistics")    
    st.dataframe(desc_stats_df)
    st.subheader("Quantile Statistics")    
    st.dataframe(quan_stats_df)

    # ---------------------------------------------------
    # Box Plot
    st.header("Box Plot", divider="orange")
    ua.show_box_plot([cleaned_df], ["Cleaned Dataset"])

    # ---------------------------------------------------
    # Distribution
    st.header("Distribution", divider="orange")
    ua.show_kde_distribution([cleaned_df], ["Cleaned Dataset"])

# ==================================================================================================
# MAIN
# ==================================================================================================
# Title
st.title("EDA - Univariate Analysis") 

# Univariate Analysis
dp_list = os.listdir("datasets/cleaned_dataset")
if(len(dp_list)>0):
    univariate_analysis()
else:
    st.warning("You need to preprocess your dataset before exploring more data insights")