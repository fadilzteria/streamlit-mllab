import random
import math
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scipy.stats import shapiro, levene, chi2_contingency # Normality Test
import pingouin as pg # Welch's ANOVA

import streamlit as st

COLORS = list(mcolors.XKCD_COLORS.keys())
random.Random(1).shuffle(COLORS)

@st.cache_data
def binary_cat_to_numeric(df):
    new_df = df.copy()
    binary_cat_columns = [col for col in df.columns if df[col].nunique()==2]

    for col in binary_cat_columns:
        first_binary, second_binary = new_df[col].unique()
        new_df[col] = new_df[col].replace({first_binary: 0, second_binary: 1})
        new_df.rename(columns={col:f'{col}_{second_binary}'}, inplace=True)

    return new_df

@st.cache_data
def extract_datetime_features(df, datetime_columns):
    date_df = pd.DataFrame()

    for col in datetime_columns:
        # Date
        date_df[f"{col}_Year"] = df[col].dt.year
        date_df[f"{col}_Quarter"] = df[col].dt.quarter
        date_df[f"{col}_Month"] = df[col].dt.month
        date_df[f"{col}_Day"] = df[col].dt.day
        date_df[f"{col}_Weekday"] = df[col].dt.weekday
        date_df[f"{col}_IsWeekend"] = (date_df[f"{col}_Weekday"] >= 5).astype(int)
        day_of_week = ["Mon", "Tues", "Wed", "Thurs", "Fri", "Sat", "Sun"]
        day_of_week = dict(enumerate(day_of_week))
        date_df[f"{col}_Weekday"] = date_df[f"{col}_Weekday"].map(day_of_week)

        # Time
        date_df[f"{col}_Time"] = df[col].dt.time
        if date_df[f"{col}_Time"].nunique()!=1:
            date_df[f"{col}_Hour"] = df[col].dt.hour
            date_df[f"{col}_Minute"] = df[col].dt.minute
            date_df[f"{col}_IsNight"] = (
                (date_df[f"{col}_Hour"] <= 6) | (date_df[f"{col}_Hour"] >= 18)
            ).astype(int)
        date_df = date_df.drop(f"{col}_Time", axis=1)

    date_df = df.merge(date_df, left_index=True, right_index=True)
    return date_df

@st.cache_data
def preprocess_data(
    df, time=False, drop_features=None,
    drop_many_cat_unique=True, max_cat_unique=10, cat_drop_first=True, scaling=True):

    # Extract Datetime Features
    extracted_df = df.copy()
    if time:
        datetime_columns = list(
            extracted_df.select_dtypes(include=['datetime64[ns]']).columns.values
        )
        extracted_df = extract_datetime_features(extracted_df, datetime_columns)
        extracted_df = extracted_df.drop(datetime_columns, axis=1)

    # Drop Features
    if drop_features is not None:
        for feat in drop_features:
            if feat in extracted_df.columns:
                extracted_df = extracted_df.drop(feat, axis=1)

    # Filter Categorical Columns
    cat_sample_df = extracted_df.copy()
    if drop_many_cat_unique:
        categorical_columns = list(
            extracted_df.select_dtypes(include=['object', 'bool']).columns.values
        )
        many_categorical_columns = [
            col for col in categorical_columns if extracted_df[col].nunique() > max_cat_unique
        ]
        cat_sample_df = cat_sample_df.drop(many_categorical_columns, axis=1)

    # Categorical Encoding
    categorical_columns = list(
        cat_sample_df.select_dtypes(include=['object', 'bool']).columns.values
    )
    if cat_drop_first:
        encoder = OneHotEncoder(
            sparse_output=False, drop='first', handle_unknown='infrequent_if_exist'
        )
    else:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist')

    ohe_features = encoder.fit_transform(cat_sample_df[categorical_columns])
    cat_feature_names = encoder.get_feature_names_out(categorical_columns)
    ohe_features = pd.DataFrame(ohe_features, columns=cat_feature_names)

    cat_sample_df = pd.concat([cat_sample_df, ohe_features], axis=1)
    cat_sample_df = cat_sample_df.drop(categorical_columns, axis=1)
    cat_sample_df[cat_feature_names] = cat_sample_df[cat_feature_names].astype(int)

    # Standardization
    if scaling:
        scaler = StandardScaler()
        scaled_sample_df = scaler.fit_transform(cat_sample_df)
    else:
        scaled_sample_df = cat_sample_df.copy()

    return df, cat_sample_df, scaled_sample_df

@st.cache_data
def extract_correlation_matrix(ori_df):
    df = copy.deepcopy(ori_df)
    df = binary_cat_to_numeric(df)

    _, cleaned_df, _ = preprocess_data(df, cat_drop_first=False, scaling=False)

    corr_columns = cleaned_df.columns.tolist()
    new_columns = set(corr_columns) - set(ori_df.columns)

    # Correlation Coefficient
    pearson_corr_matrix = cleaned_df.corr()
    spearman_corr_matrix = cleaned_df.corr(method='spearman')
    kendall_corr_matrix = cleaned_df.corr(method='kendall')
    all_matrices = [pearson_corr_matrix, spearman_corr_matrix, kendall_corr_matrix]

    # Create Dataframe
    corr_matrix_df = pd.DataFrame()
    for i in range(len(corr_columns)):
        for j in range(i+1, len(corr_columns)):
            temp_corr_df = pd.DataFrame()
            temp_corr_df.loc[0, "Column 1"] = pearson_corr_matrix.index[i]
            temp_corr_df.loc[0, "Column 2"] = pearson_corr_matrix.columns[j]
            temp_corr_df.loc[0, "Pearson Corr"] = pearson_corr_matrix.iloc[i, j]
            temp_corr_df.loc[0, "Spearman Corr"] = spearman_corr_matrix.iloc[i, j]
            temp_corr_df.loc[0, "Kendall Corr"] = kendall_corr_matrix.iloc[i, j]

            corr_matrix_df = pd.concat([corr_matrix_df, temp_corr_df], ignore_index=True)

    return corr_matrix_df, all_matrices, new_columns

@st.cache_data
def show_correlation_heatmap(all_matrices, corr_type):
    # Matrix
    if corr_type=="Pearson":
        matrix = all_matrices[0]
    elif corr_type=="Spearman":
        matrix = all_matrices[1]
    else:
        matrix = all_matrices[2]

    # Create Subplots
    fig, ax = plt.subplots(figsize=(15, 12))

    # Heatmap
    _ = sns.heatmap(
        matrix, cmap='RdYlGn', vmin=-1.0, vmax=1.0, linewidths=0.5, linecolor='White', square=True
    )

    # Update Axes
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=-40, ha="right", rotation_mode="anchor")

    # Show
    plt.tight_layout()
    plt.title(f"{corr_type} Correlation")
    st.pyplot(fig)

@st.cache_data
def extract_strong_weak_corr_matrix(
    corr_df, new_columns, corr_type, max_corr=None,
    pos_value=0.6, neg_value=-0.6, min_zero_value=-0.1, max_zero_value=0.1):
    # Filtering
    corr_df = corr_df[~corr_df["Column 1"].isin(new_columns)].reset_index(drop=True)
    corr_df = corr_df[~corr_df["Column 2"].isin(new_columns)].reset_index(drop=True)

    # Strong Positive
    pos_corr_df = corr_df[corr_df[f"{corr_type} Corr"] >= pos_value].reset_index(drop=True)
    # Strong Negative
    neg_corr_df = corr_df[corr_df[f"{corr_type} Corr"] <= neg_value].reset_index(drop=True)
    # Weak/No Correlation
    no_corr_df = corr_df[
        (corr_df[f"{corr_type} Corr"] >= min_zero_value)
        &(corr_df[f"{corr_type} Corr"] <= max_zero_value)
    ].reset_index(drop=True)

    # Sort Values
    pos_corr_df = pos_corr_df.sort_values(
        by=f"{corr_type} Corr", ascending=False
    ).reset_index(drop=True)
    neg_corr_df = neg_corr_df.sort_values(
        by=f"{corr_type} Corr", ascending=True
    ).reset_index(drop=True)
    no_corr_df = no_corr_df.sort_values(
        by=f"{corr_type} Corr", ascending=True, key=abs
    ).reset_index(drop=True)

    if max_corr is not None:
        pos_corr_df = pos_corr_df.iloc[:max_corr]
        neg_corr_df = neg_corr_df.iloc[:max_corr]
        no_corr_df = no_corr_df.iloc[:max_corr]

    strong_weak_corrs = [pos_corr_df, neg_corr_df, no_corr_df]
    corr_titles = [
        f"Strong Positive {corr_type} Correlation",
        f"Strong Negative {corr_type} Correlation",
        f"Weak/No {corr_type} Correlation",
    ]

    return strong_weak_corrs, corr_titles

@st.cache_data
def extract_correlation_series(df, feature, corr_type):
    # Columns
    numerical_columns = list(df.select_dtypes(include=[np.number]).columns.values)
    if feature in numerical_columns:
        numerical_columns.remove(feature)
    num_binary_df = df[numerical_columns]

    # Correlation Coefficient with One Feature
    if df[feature].dtypes in ["int64", "float64"]:
        series = df[feature]
    else:
        series = pd.Series(df.index)
    pearson_corr_series = num_binary_df.corrwith(series)
    spearman_corr_series = num_binary_df.corrwith(series, method='spearman')
    kendall_corr_series = num_binary_df.corrwith(series, method='kendall')

    corr_series_df = pd.DataFrame({
        "Column 1": [feature]*len(pearson_corr_series),
        "Column 2": pearson_corr_series.index,
        'Pearson Corr': pearson_corr_series.values,
        'Spearman Corr': spearman_corr_series.values,
        'Kendall Corr': kendall_corr_series.values
    })

    corr_series_df = corr_series_df.sort_values(
        by=f"{corr_type} Corr", ascending=False, key=abs
    ).reset_index(drop=True)

    return corr_series_df

@st.cache_data
def show_correlation_scatter_plot(df, corr_dfs, corr_titles):
    for i, (corr_df, corr_title) in enumerate(zip(corr_dfs, corr_titles)):
        if corr_df.shape[0]==0:
            print(f"No {corr_title}")
        else:
            # Create Subplots
            num_columns = corr_df.shape[0]
            nrows, ncols = ((num_columns-1)//4)+1, 4
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows*4))

            # For Each Column
            for j in range(corr_df.shape[0]):
                col_name_1 = corr_df.loc[j, "Column 1"]
                col_name_2 = corr_df.loc[j, "Column 2"]
                pearson_value = round(corr_df.loc[j, "Pearson Corr"], 3)
                spearman_value = round(corr_df.loc[j, "Spearman Corr"], 3)
                kendall_value = round(corr_df.loc[j, "Kendall Corr"], 3)

                value_1 = df[col_name_1] if df[col_name_1].dtypes!="object" else pd.Series(df.index)
                value_2 = df[col_name_2] if df[col_name_2].dtypes!="object" else pd.Series(df.index)

                if nrows==1:
                    ax_temp = axs[j]
                else:
                    ax_temp = axs[j//ncols, j%ncols]

                # Scatter Plots
                ax_temp.scatter(
                    value_1, value_2,
                    c=COLORS[i], s=8, alpha=0.8
                )
                ax_temp.set_title(
                    f"Pearson: {pearson_value:.5f}, Spearman: {spearman_value:.5f}, \
                    Kendall: {kendall_value:.5f}", size=7
                )
                ax_temp.set_xlabel(col_name_1)
                ax_temp.set_ylabel(col_name_2)

            if j < (nrows*ncols):
                for k in range(j+1, (nrows*ncols)):
                    if nrows==1:
                        ax_temp = axs[k]
                    else:
                        ax_temp = axs[k//ncols, k%ncols]
                    ax_temp.axis("off")

            plt.suptitle(corr_title)
            plt.tight_layout()
            st.pyplot(fig)

@st.cache_data
def test_numerical_categorical(
    df, sample_size=30, max_unique=5,
    min_sample_mean=20, max_sample_mean=None, feature=None):
    categorical_columns = list(df.select_dtypes(include=['object', 'bool']).columns.values)
    numerical_columns = list(df.select_dtypes(include=[np.number]).columns.values)
    if feature:
        if df[feature].dtypes in ["object", "bool"]:
            categorical_columns = [feature]
        else:
            numerical_columns = [feature]

    bi_idx = 0
    result_df = pd.DataFrame()
    all_sample_group_means = []

    # For Each Categorical Columns
    for _, cat_col in enumerate(categorical_columns):
        # Check Number of Unique Values that are Qualified for Statistical Inference
        # ------------------------------------------------
        value_df = df[cat_col].value_counts().reset_index()
        value_df = value_df[value_df["count"]>=(sample_size*min_sample_mean)].head(max_unique)
        if len(value_df) < 2:
            continue
        print(cat_col)

        # Get Group Dataframe and Indexes
        # -------------------------------
        group_dfs = []
        group_idxs = []

        # For Each Group in Unique Values
        for group in value_df[cat_col].tolist():
            # Dataframe
            group_df = df[df[cat_col]==group].reset_index(drop=True)
            group_dfs.append(group_df)

            # Get Sample Indexes
            if max_sample_mean is None:
                max_idx = len(group_df)
            else:
                max_idx = min(sample_size*max_sample_mean, len(group_df))
            idxs = np.arange(0, max_idx, sample_size)
            if idxs[-1] != (len(group_df)-sample_size):
                idxs = idxs[:-1]
            group_idxs.append(idxs)

        # For Each Numerical Columns
        for _, num_col in enumerate(numerical_columns):
            print(f"- {num_col}")
            result_df.loc[bi_idx, "Cat Column"] = cat_col
            result_df.loc[bi_idx, "Num Column"] = num_col

            # Get Sample Means for Each Group
            # -------------------------------
            sample_group_means = []

            # For Each Group in Unique Values
            for g, group in enumerate(value_df[cat_col].tolist()):
                # Get Sample Means
                group_df = group_dfs[g].sample(frac=1, random_state=42, ignore_index=True)
                sample_means = []
                for idx in group_idxs[g]:
                    sample_mean = group_df[num_col][idx:idx+sample_size].mean()
                    sample_means.append(sample_mean)

                sample_group_means.append((group, sample_means))

            all_sample_group_means.append(sample_group_means)

            # Check if Unique Values are used Fully or Partial
            result_df.loc[bi_idx, "Num Groups"] = len(sample_group_means)
            if len(sample_group_means)==df[cat_col].nunique():
                result_df.loc[bi_idx, "Full Groups"] = "Full"
            else:
                result_df.loc[bi_idx, "Full Groups"] = "Partial"

            # Normality Test
            # --------------
            normal = True
            for sample_means in sample_group_means:
                group, sample_means = sample_means[0], sample_means[1]
                _, norm_p_value = shapiro(sample_means) # Shapiro-Wilk Test
                if norm_p_value < 0.05:
                    normal = False
                    break

            if normal is False:
                result_df.loc[bi_idx, "Normality"] = "Not Normal"
                bi_idx += 1
                continue
            else:
                result_df.loc[bi_idx, "Normality"] = "Normal"

            # Homogeneity Test with Levene Test
            # ---------------------------------
            _, homogeneity_p_value = levene(
                *(sample_means[1] for sample_means in sample_group_means), center='mean'
            )
            result_df.loc[bi_idx, "p-value Homogeneity"] = homogeneity_p_value

            # ANOVA Test
            # ----------
            all_sample_means = []
            all_sample_groups = []
            for sample_means in sample_group_means:
                all_sample_means.extend(sample_means[1])
                all_sample_groups.extend([sample_means[0]]*len(sample_means[1]))

            anova_df = pd.DataFrame({
                'score': all_sample_means,
                'group': all_sample_groups
            })

            if homogeneity_p_value < 0.05: # Unequal Variance => Welch's ANOVA
                result_df.loc[bi_idx, "Homogeneity"] = "Unequal"
                an_result = pg.welch_anova(dv='score', between='group', data=anova_df)

            else: # Equal Variance => ANOVA
                result_df.loc[bi_idx, "Homogeneity"] = "Equal"
                an_result = pg.anova(dv='score', between='group', data=anova_df)

            anova_p_value = an_result.loc[0, "p-unc"]
            result_df.loc[bi_idx, "ANOVA p-value"] = anova_p_value
            result_df.loc[bi_idx, "Partial Eta-squared"] = an_result.loc[0, "np2"]

            # Check Significance
            # ------------------
            if anova_p_value < 0.05:
                result_df.loc[bi_idx, "Conclusion"] = "Rejected"
            else:
                result_df.loc[bi_idx, "Conclusion"] = "Failed to Reject"

            bi_idx += 1

    return result_df, all_sample_group_means

@st.cache_data
def show_correlation_num_cat(df_result, all_group_means, num_groups=5, num_plots=5):
    if len(df_result)==0 or "Conclusion" not in df_result.columns:
        print("No Correlation between Numerical and Categorical")
        return

    # Dataframe
    rejected_df_result = df_result[
        (df_result["Normality"]=="Normal")&(df_result["Num Groups"] <= num_groups)
    ]
    rejected_df_result = rejected_df_result.sort_values(
        by="Partial Eta-squared", ascending=False
    ).iloc[:num_plots]
    num_plots = len(rejected_df_result)

    # Create Subplots
    fig, axs = plt.subplots(num_plots, 2, figsize=(15, num_plots*4))

    for i, (idx, row) in enumerate(rejected_df_result.iterrows()):
        groups = [sample_means[0] for sample_means in all_group_means[idx]]
        list_data = [sample_means[1] for sample_means in all_group_means[idx]]
        # print(groups)

        # Violin Plot
        ax_temp = axs[0] if num_plots==1 else axs[i, 0]
        violin_parts = ax_temp.violinplot(list_data, vert=False)
        ax_temp.set_title(
            f"Variance: {row['Homogeneity']}, p-Value: {row['ANOVA p-value']:.4f}, \
            Partial Eta-squared: {row['Partial Eta-squared']:.4f}", size=8
        )
        ax_temp.set(ylabel=row["Cat Column"])
        ax_temp.set(xlabel=row["Num Column"])
        ax_temp.set_yticks(np.arange(1, len(groups) + 1))
        ax_temp.set_yticklabels(groups)
        for pc, color in zip(violin_parts['bodies'], COLORS[:len(groups)]):
            pc.set_facecolor(color)

        # Distribution
        ax_temp = axs[1] if num_plots==1 else axs[i, 1]
        for j, group_means in enumerate(list_data):
            sns.kdeplot(group_means, color=COLORS[j], fill=True, label=groups[j], ax=ax_temp)
        ax_temp.set_title(row["Num Column"], size=8)
        ax_temp.legend(loc="upper right")

    plt.suptitle("The Correlation between Numerical and Categorical Columns", y=1)
    plt.tight_layout()
    st.pyplot(fig)

@st.cache_data
def test_two_categorical(df, max_unique=20, feature=None):
    categorical_columns = list(df.select_dtypes(include=['object', 'bool']).columns.values)
    result_df = pd.DataFrame()
    bi_idx = 0

    # For Each Paired Categorical Columns
    few_categorical_columns = [
        col for col in categorical_columns if df[col].nunique() <= max_unique
    ]
    if feature:
        few_categorical_columns.remove(feature)
    for i, column_1 in enumerate(few_categorical_columns):
        if feature:
            few_categorical_columns_2 = [feature]
        else:
            few_categorical_columns_2 = few_categorical_columns[i+1:]

        for _, column_2 in enumerate(few_categorical_columns_2):
            result_df.loc[bi_idx, "Cat Column 1"] = column_1
            result_df.loc[bi_idx, "Cat Column 2"] = column_2

            # Contingency Table
            contingency_table = pd.crosstab(df[column_1], df[column_2])

            # Chi-squared Test
            statistic, pvalue, _, expected_freq = chi2_contingency(contingency_table)
            result_df.loc[bi_idx, "Chi-squared p-value"] = pvalue

            # Expected Cells
            expected_cell_one = (expected_freq < 1).sum().sum()
            expected_cell_percentages = (expected_freq >= 5).sum().sum() / \
                (expected_freq.shape[0]*expected_freq.shape[1])
            expected_cell_percentages *= 100
            result_df.loc[bi_idx, "Expected Cell Percentages"] = round(expected_cell_percentages, 2)

            # Expected Cells Must be Greater than or Equal 80% and Not Less Than 1
            if expected_cell_percentages >= 80 and expected_cell_one is False:
                result_df.loc[bi_idx, "Assumptions"] = "Success"

                # Effect Size
                if df[column_1].nunique()==2 and df[column_2].nunique()==2:
                    # Phi Coefficient for Binary Variables
                    a, b = contingency_table.iloc[0]
                    c, d = contingency_table.iloc[1]
                    phi_v = (a*d-b*c) / math.sqrt((a+b)*(c+d)*(a+c)*(b+d))
                    result_df.loc[bi_idx, "Variables"] = "Two Binary"
                    result_df.loc[bi_idx, "Phi Coefficient"] = phi_v
                else:
                    # Cramer's V Value for Nominal Variables
                    cramer_v = math.sqrt((statistic/df.shape[0])/(min(contingency_table.shape)-1))
                    result_df.loc[bi_idx, "Variables"] = "At Least 1 Nominal"
                    result_df.loc[bi_idx, "Cramer's Value"] = cramer_v

            else:
                result_df.loc[bi_idx, "Assumptions"] = "Failed"

            bi_idx += 1

    return result_df

@st.cache_data
def show_correlation_two_binary(df, df_result, num_two_binary=6):
    # Check Phi Coefficient
    if "Phi Coefficient" not in df_result.columns:
        print("No Correlation between Two Binary Variables")
        return

    # Dataframe
    two_binary_df = df_result[df_result["Variables"]=="Two Binary"]
    two_binary_df = two_binary_df.sort_values(
        by="Phi Coefficient", ascending=False, key=abs
    ).iloc[:num_two_binary]
    num_two_binary = len(two_binary_df)

    # Create Subplots
    nrows, ncols = ((num_two_binary-1)//3)+1, 3
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows*4))

    for i, (_, row) in enumerate(two_binary_df.iterrows()):
        column_1 = row["Cat Column 1"]
        column_2 = row["Cat Column 2"]
        ax_temp = axs[i] if nrows==1 else axs[i//3, i%3]

        # Color and Crossbar
        contingency_table = pd.crosstab(df[column_1], df[column_2])
        contingency_table_norm = pd.crosstab(df[column_1], df[column_2], normalize="index")
        contingency_table_norm *= 100

        # Show in the Visualization
        contingency_table_viz = contingency_table.astype(str) + "\n" + \
            round(contingency_table_norm, 2).astype(str) + "%"

        # Heatmap
        _ = sns.heatmap(
            contingency_table_norm, annot=contingency_table_viz, fmt="", annot_kws={"fontsize": 10},
            ax=ax_temp, cmap='RdYlGn', vmin=0, vmax=100, linewidths=0.5, linecolor='White',
            square=True, cbar=False
        )

        # Update Axes
        ax_temp.set_title(
            f"Chi-squared p-Value: {row['Chi-squared p-value']:.4f}, \
            Phi Coefficient: {row['Phi Coefficient']:.4f}", size=8
        )

    if (len(two_binary_df) - 1) < (nrows*ncols):
        for j in range(len(two_binary_df), (nrows*ncols)):
            ax_temp = axs[j] if nrows==1 else axs[j//3, j%3]
            ax_temp.axis("off")

    plt.suptitle("Correlations between Two Binary")
    plt.tight_layout()
    st.pyplot(fig)

@st.cache_data
def show_correlation_least_one_nominal(df, df_result, num_one_nominal=5):
    # Check Cramer's Value
    if "Cramer's Value" not in df_result.columns:
        print("No Correlation between Two Categorical Variables")
        return

    # Dataframe
    one_nominal_df = df_result[df_result["Variables"]=="At Least 1 Nominal"]
    one_nominal_df = one_nominal_df.sort_values(
        by="Cramer's Value", ascending=False, key=abs
    ).iloc[:num_one_nominal]
    num_one_nominal = len(one_nominal_df)

    # Create Subplots
    fig, axs = plt.subplots(num_one_nominal, 1, figsize=(15, num_one_nominal*4))

    for i, (_, row) in enumerate(one_nominal_df.iterrows()):
        column_1 = row["Cat Column 1"]
        column_2 = row["Cat Column 2"]
        ax_temp = axs if num_one_nominal==1 else axs[i]

        # Color and Crossbar
        if df[column_1].nunique() > df[column_2].nunique():
            column_1, column_2 = column_2, column_1
        contingency_table = pd.crosstab(df[column_1], df[column_2])
        contingency_table_norm = pd.crosstab(df[column_1], df[column_2], normalize="index")
        contingency_table_norm *= 100

        # Show in the Visualization
        contingency_table_viz = contingency_table.astype(str) + "\n" + \
            round(contingency_table_norm, 2).astype(str) + "%"

        # Heatmap
        _ = sns.heatmap(
            contingency_table_norm, annot=contingency_table_viz, fmt="", annot_kws={"fontsize": 8},
            ax=ax_temp, cmap='RdYlGn', vmin=0, vmax=100, linewidths=0.5, linecolor='White',
            square=True, cbar=False
        )

        # Update Axes
        ax_temp.set_title(
            f"""Chi-squared p-Value: {row['Chi-squared p-value']:.4f}, \
            Cramer's Value: {row["Cramer's Value"]:.4f}""", size=8
        )

    plt.suptitle("Correlations between Two Categorical At Least One Nominal", y=1.00)
    plt.tight_layout()
    st.pyplot(fig)
