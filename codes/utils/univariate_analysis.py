import random
import math
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import streamlit as st

COLORS = list(mcolors.XKCD_COLORS.keys())
random.Random(1).shuffle(COLORS)

@st.cache_data
def calculate_value_counts(df, just_category=False, drop_features=None, max_numeric_uniques=20):
    # Categorical and Few Numerical Columns
    if drop_features is not None:
        for feat in drop_features:
            if feat in df.columns:
                df = df.drop(feat, axis=1)

    categorical_columns = list(df.select_dtypes(include=['object', 'bool']).columns.values)
    numerical_columns = list(df.select_dtypes(include=[np.number]).columns.values)
    few_num_columns = [col for col in numerical_columns if df[col].nunique() <= max_numeric_uniques]
    if just_category:
        value_columns = categorical_columns
    else:
        value_columns = categorical_columns + few_num_columns

    # For Each Column
    value_df_list = []
    for _, col in enumerate(value_columns):
        # Dataframe
        value_df = df[col].value_counts().reset_index()
        col_count = "count"

        # Sort Values
        if col in categorical_columns:
            value_df = value_df.sort_values(by=col_count, ascending=False).reset_index(drop=True)
        elif col in few_num_columns:
            value_df = value_df.sort_values(by=col, ascending=True).reset_index(drop=True)

        # Calculate Percentage
        value_df["count (%)"] = round((value_df["count"] / df.shape[0]) * 100, 3)

        value_df_list.append(value_df)

    return value_df_list

@st.cache_data
def show_value_counts(full_list, labels, max_cat_uniques=10):
    # Check List
    if len(full_list[0])==0:
        print("No Value Counts")
    else:
        # Create Subplots
        nrows, ncols = len(full_list[0]), len(full_list)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows*3))

        # For Each Label
        for j, (value_df_list, label) in enumerate(zip(full_list, labels)):
            # For Each Column
            for i, value_df in enumerate(value_df_list):
                ax_temp = axs[i] if ncols==1 else axs[i, j]

                col_name = value_df.columns.tolist()[0]
                col_perc = value_df.columns.tolist()[2]
                var = value_df.iloc[0, 0]

                if isinstance(var, (int, np.integer, float)): # Numerical
                    # Vertical Bar Plot
                    ax_temp.bar(
                        value_df[col_name].astype(str), value_df[col_perc],
                        align='center', width=0.5, color=COLORS[j], label=label
                    )
                    ax_temp.set_ylim(0, 110)

                else: # Categorical
                    value_df = value_df.iloc[:max_cat_uniques, :]

                    if isinstance(var, object): # Boolean
                        col_name_list = [str(x) for x in value_df[col_name].values]
                    else:
                        col_name_list = value_df[col_name].values

                    # Horizontal Sorted Bar Plot
                    ax_temp.barh(
                        col_name_list, value_df[col_perc],
                        align='center', height=0.5, color=COLORS[j], label=label
                    )

                    # Text
                    for k, v in enumerate(value_df[col_perc]):
                        ax_temp.text(v+0.8, k, str(v)+"%", va='center', fontsize=9)

                    # Update Axes
                    ax_temp.invert_yaxis()  # Read Columns Top-to-Bottom
                    ax_temp.tick_params(axis='y', which='major', pad=15, labelsize=10)
                    ax_temp.tick_params(left=False, bottom=True)
                    ax_temp.set_xlim(0, 108)

                sns.despine(top=True, right=True)
                ax_temp.set_title(f"{col_name}: {label}", size=12)

        plt.suptitle("Value Counts", y=1.00)
        plt.tight_layout()
        st.pyplot(fig)

@st.cache_data
def extract_statistics(df):
    numerical_columns = list(df.select_dtypes(include=[np.number]).columns.values)

    # Descriptive Statistics
    desc_stats_df = pd.DataFrame(numerical_columns, columns=["Column"])
    desc_stats_df["Mean"] = df[numerical_columns].mean().values
    desc_stats_df["Sum"] = df[numerical_columns].sum().values
    desc_stats_df["Var"] = df[numerical_columns].var().values
    desc_stats_df["Std"] = df[numerical_columns].std().values
    desc_stats_df["Skewness"] = df[numerical_columns].skew().values
    desc_stats_df["Kurtosis"] = df[numerical_columns].kurt().values

    # Quantile Statistics
    quan_stats_df = pd.DataFrame(numerical_columns, columns=["Column"])
    quan_stats_df["Min"] = df[numerical_columns].min().values
    quan_stats_df["5%"] = df[numerical_columns].quantile(0.05).values
    quan_stats_df["Q1"] = df[numerical_columns].quantile(0.25).values
    quan_stats_df["Median"] = df[numerical_columns].median().values
    quan_stats_df["Q3"] = df[numerical_columns].quantile(0.75).values
    quan_stats_df["95%"] = df[numerical_columns].quantile(0.95).values
    quan_stats_df["Max"] = df[numerical_columns].max().values
    quan_stats_df["Range"] = quan_stats_df["Max"] - quan_stats_df["Min"]
    quan_stats_df["IQR"] = quan_stats_df["Q3"] - quan_stats_df["Q1"]

    return desc_stats_df, quan_stats_df

@st.cache_data
def show_box_plot(dfs, labels):
    # Columns
    for i, df in enumerate(dfs):
        numerical_columns = list(df.select_dtypes(include=[np.number]).columns.values)
        if i==0:
            inner_numerical_columns = set(numerical_columns)
        else:
            inner_numerical_columns = inner_numerical_columns.intersection(numerical_columns)
    inner_numerical_columns = list(inner_numerical_columns)

    # Create Subplots
    len_cols = len(inner_numerical_columns)
    nrows, ncols = ((len_cols-1)//3)+1, 3
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(15, nrows*int(math.ceil(len(dfs)*2)))
    )

    # Box Plots
    for i, col in enumerate(inner_numerical_columns):
        list_df = [copy.deepcopy(df[col]) for df in dfs]
        list_df = [df.dropna() for df in list_df]
        ax_temp = axs[i] if nrows==1 else axs[i//ncols, i%ncols]

        ax_temp.boxplot(list_df, vert=False)
        ax_temp.set_title(col, size=12)
        ax_temp.set(xlabel=None)
        ax_temp.set(ylabel=None)
        ax_temp.tick_params(left=False)
        ax_temp.set(yticklabels=labels) if i%ncols==0 else ax_temp.set(yticklabels=[])

    if i < (nrows*ncols):
        for j in range(i+1, (nrows*ncols)):
            ax_temp = axs[j] if nrows==1 else axs[j//ncols, j%ncols]
            ax_temp.axis("off")

    plt.suptitle("Box Plot", y=1)
    plt.tight_layout()
    st.pyplot(fig)

@st.cache_data
def show_kde_distribution(dfs, labels):
    # Columns
    for i, df in enumerate(dfs):
        numerical_columns = list(df.select_dtypes(include=[np.number]).columns.values)
        if i==0:
            inner_numerical_columns = set(numerical_columns)
        else:
            inner_numerical_columns = inner_numerical_columns.intersection(numerical_columns)
    inner_numerical_columns = list(inner_numerical_columns)

    # Create Subplots
    len_cols = len(inner_numerical_columns)
    nrows, ncols = ((len_cols-1)//4)+1, 4
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows*4))

    # KDE Plots
    for i, col in enumerate(inner_numerical_columns):
        ax_temp = axs[i] if nrows==1 else axs[i//ncols, i%ncols]

        for j, (df, label) in enumerate(zip(dfs, labels)):
            sns.kdeplot(df[col], color=COLORS[j], fill=True, label=label, ax=ax_temp)
        ax_temp.set_title(col, size=12)
        ax_temp.set(xlabel=None)
        ax_temp.set(ylabel=None)
        if i==3:
            ax_temp.legend(loc="upper right")

    if i < (nrows*ncols):
        for j in range(i+1, (nrows*ncols)):
            ax_temp = axs[j] if nrows==1 else axs[j//ncols, j%ncols]
            ax_temp.axis("off")

    plt.suptitle("KDE Distribution Plot")
    plt.tight_layout()
    st.pyplot(fig)
