import os
import pandas as pd
import streamlit as st
from codes.utils import univariate_analysis as ua

# ==================================================================================================
# UNIVARIATE ANALYSIS
# ==================================================================================================
@st.fragment()
def value_counts(cleaned_df):
    with st.container(border=True):
        col_1, col_2, col_3 = st.columns(3)
        with col_1:
            st.toggle("Only Use Categorical", value=False, key="just_category")
        with col_2:
            st.slider(
                "Max Uniques for Numerical",
                min_value=3, max_value=15, value=8, key="max_num_uniques"
            )
        with col_3:
            st.slider(
                "Top Uniques for Categorical",
                min_value=2, max_value=15, value=10, key="max_cat_uniques"
            )

    all_value_df_list = ua.extract_value_counts(
        [cleaned_df], just_category=st.session_state["just_category"],
        max_numeric_uniques=st.session_state["max_num_uniques"]
    )
    ua.show_value_counts(
        all_value_df_list, ["Cleaned Dataset"], max_cat_uniques=st.session_state["max_cat_uniques"]
    )

def univariate_analysis():
    # ---------------------------------------------------
    # Dataframe
    st.selectbox(label="Data Preprocessing Name", options=dp_list, key="dp_name", index=0)
    dp_path = os.path.join("datasets/cleaned_dataset", st.session_state["dp_name"])
    cleaned_df_path = os.path.join(dp_path, "cleaned_train.parquet")
    cleaned_df = pd.read_parquet(cleaned_df_path)

    # ---------------------------------------------------
    # Value Counts
    st.header("Value Counts", divider="orange")
    value_counts(cleaned_df)

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
if len(dp_list)>0:
    univariate_analysis()
else:
    st.warning("You need to preprocess your dataset before exploring more data insights")
