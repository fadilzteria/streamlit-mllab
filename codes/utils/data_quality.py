import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import streamlit as st

COLORS = list(mcolors.XKCD_COLORS.keys())
random.Random(1).shuffle(COLORS)

def load_dataframe(filename, uploaded_file=None):
    if uploaded_file:
        file = uploaded_file
    else:
        file = filename

    if filename[-3:]=="csv":
        df = pd.read_csv(file)
    elif filename[-3:]=="pkl":
        df = pd.read_pickle(file)
    elif filename[-7:]=="parquet":
        df = pd.read_parquet(file)
    else:
        df = pd.read_feather(file)

    return df

@st.cache_data
def check_other_values(df, task, drop_features=None, max_unique=20):
    spec_col = f"{task} Counts"

    if drop_features is not None:
        for feat in drop_features:
            if feat in df.columns:
                df = df.drop(feat, axis=1)

    if task=="Unique":
        unique_columns = [col for col in df.columns if df[col].nunique() <= max_unique]
        unique_list = [(col, df[col].nunique()) for col in unique_columns]
        other_df = pd.DataFrame(unique_list, columns=["Column", spec_col])
    elif task=="Missing":
        other_df = df.isnull().sum(axis=0).reset_index()
    else:
        df = df.select_dtypes('number')

        if task=="Infinite": # Infinite
            other_df = np.isinf(df).sum(axis=0).reset_index()
        elif task=="Zero": # Zero
            other_df = (df == 0).astype(int).sum(axis=0).reset_index()
        else: # Negative
            other_df = (df < 0).astype(int).sum(axis=0).reset_index()

    other_df.columns = ["Column", spec_col]
    other_df = other_df.sort_values(by=spec_col, ascending=False).reset_index(drop=True)
    if task!="Unique":
        other_df = other_df[other_df[spec_col]!=0].reset_index(drop=True)
        other_df[f"{spec_col} (%)"] = round((other_df[spec_col] / df.shape[0]) * 100, 3)

    return other_df

@st.cache_data
def show_other_values(dfs, labels):
    # Sum Other Values
    col = dfs[0].columns.tolist()[1]
    available_values = 0
    max_height_plot = 0
    for df in dfs:
        available_values += (df.shape[0]!=0)
        max_height_plot = max(df.shape[0], max_height_plot)

    if available_values==0:
        print(f"No {col}")
    else:
        # Create Subplots
        fig, axs = plt.subplots(nrows=1, ncols=available_values, figsize=(12, max_height_plot))

        # Vertical Sorted Bar Plot
        df = dfs[-1]
        if len(df.columns.tolist())>2: # No Unique
            col = df.columns.tolist()[2]

        k = 0
        for _, (df, label) in enumerate(zip(dfs, labels)):
            if df.shape[0]==0:
                continue

            if available_values==1:
                ax_temp = axs
            else:
                ax_temp = axs[k]

            ax_temp.barh(
                df["Column"], df[col],
                align='center', height=0.5, color=COLORS[k], label=label
            )

            if len(df.columns.tolist())>2: # No Unique
                # Text
                for j, v in enumerate(df[col]):
                    ax_temp.text(v+0.8, j, str(v)+'%', va='center', size=10)

                # Update Axes
                ax_temp.set_xlim(0, 108)
                sns.despine(top=True, right=True)
            else: # Unique
                # Text
                for j, v in enumerate(df[col]):
                    ax_temp.text(v+0.8, j, str(v), va='center', size=10)

                # Update Axes
                ax_temp.tick_params(left=False, bottom=False)
                ax_temp.set_xticklabels([])
                sns.despine(top=True, right=True, left=True, bottom=True)

            ax_temp.invert_yaxis()  # Read Columns Top-to-Bottom
            ax_temp.tick_params(axis='y', which='major', pad=10, labelsize=10)
            ax_temp.set_title(f"{label}: {df.shape[0]} Features", size=10)

            k += 1

        plt.suptitle(col, y=1.00)
        plt.tight_layout()
        st.pyplot(fig)

@st.cache_data
def drop_unique_features(df, persisted_features=None):
    # Drop Full Unique Features
    full_unique_features = [col for col in df.columns if df[col].nunique() == len(df)]

    if persisted_features is not None:
        for feat in persisted_features:
            if feat in full_unique_features:
                full_unique_features.remove(feat)

    if len(full_unique_features)>0:
        print("Drop Full Unique Features")
        for feature in full_unique_features:
            print(f"-> {feature}")
            df = df.drop(feature, axis=1)

    # Drop Uniform Features
    uniform_features = [col for col in df.columns if df[col].nunique() == 1]

    if len(persisted_features) > 0:
        for feat in persisted_features:
            if feat in uniform_features:
                uniform_features.remove(feat)

    if len(uniform_features)>0:
        print("Drop Uniform Features")
        for feature in uniform_features:
            print(f"-> {feature}")
            df = df.drop(feature, axis=1)

    return df

@st.cache_data
def check_duplicated_rows(df, drop_duplicated_rows=False, target_exist=None, spec_features=None):
    # Dataframe All Columns
    dup_all = df[df.duplicated(keep=False)]
    st.write("Duplicated Rows for Dataframe All Columns:", dup_all.shape[0])

    if dup_all.shape[0]!=0:
        if drop_duplicated_rows:
            df = df.drop_duplicates()

        else:
            st.write(f"Shape: {dup_all.shape}")
            st.dataframe(dup_all)

    # Dataframe without Target and Specific Features
    if target_exist:
        df_no_spec_feat = df.drop(spec_features, axis=1) if len(spec_features) > 0 else df.copy()
        dup_columns = df_no_spec_feat.columns.tolist()
        dup_columns.remove(target_exist)
        df_no_feat_target = df_no_spec_feat.sort_values(by=dup_columns)

        # Extract Index
        last_dup_no_feat_target = df_no_feat_target[
            df_no_feat_target.duplicated(subset=dup_columns)
        ]
        dup_index = last_dup_no_feat_target.drop_duplicates(
            keep="last", subset=dup_columns
        ).index.tolist()

        # Get All Duplicates
        dup_no_feat_target = df_no_feat_target[
            df_no_feat_target.duplicated(keep=False, subset=dup_columns)
        ]
        st.write(
            "Duplicated Rows for Dataframe Without Specific Features and Target:",
            dup_no_feat_target.shape[0]
        )

        if dup_no_feat_target.shape[0]!=0:
            if drop_duplicated_rows:
                index_list = dup_no_feat_target.index.tolist()

                if df[target_exist].dtypes in ["object", "bool"]: # Object
                    dup_group_no_feat_target = dup_no_feat_target.groupby(
                        dup_columns, dropna=False
                    )[target_exist].mode().reset_index()
                else: # Numeric
                    dup_group_no_feat_target = dup_no_feat_target.groupby(
                        dup_columns, dropna=False
                    )[target_exist].mean().reset_index()
                dup_group_no_feat_target[spec_features[0]] = dup_index

                df = df[~df.index.isin(index_list)]
                df = pd.concat([df, dup_group_no_feat_target], ignore_index=True)
                df = df.sort_values(by=spec_features[0]).reset_index(drop=True)
            else:
                st.write(f"Shape: {dup_no_feat_target.shape}")
                st.dataframe(dup_no_feat_target)

    return df

@st.cache_data
def transform_dtypes(df, split="Train", num2bin_cols=None, num2cat_cols=None):
    # Numerical to Binary
    if split=="Train":
        num_columns = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]
        num2bin_cols = {}
        for col in num_columns:
            if df[col].nunique()==2:
                uniques = df[col].unique()
                uniques = uniques[~np.isnan(uniques)]
                if (np.sort(uniques) == [0, 1]).all():
                    num2bin_cols[col] = [0, 1]
                    df[col] = df[col].astype(bool)
                else:
                    first_binary, second_binary = uniques
                    num2bin_cols[col] = [first_binary, second_binary]
                    df[col] = df[col].replace({first_binary: False, second_binary: True})
                    df.rename(columns={col:f'{col}_{second_binary}'}, inplace=True)

    else:
        for col in num2bin_cols:
            binaries = num2bin_cols[col]
            if binaries == [0, 1]:
                df[col] = df[col].astype(bool)
            else:
                first_binary, second_binary = uniques
                df[col] = df[col].replace({first_binary: False, second_binary: True})
                df.rename(columns={col:f'{col}_{second_binary}'}, inplace=True)

    # Numerical to Categorical
    if num2cat_cols is not None:
        df[num2cat_cols] = df[num2cat_cols].apply(lambda x: x.astype(str), axis=0)

    # Object to Datetime
    object_columns = list(df.select_dtypes(include=['object']).columns.values)
    pattern = r'(\d{2,4}(-|/)\d{2}(-|/)\d{2,4})+'
    for col in object_columns:
        if df[col].str.match(pattern).all():
            df[col] = pd.to_datetime(df[col])

    return df, num2bin_cols

@st.cache_data
def impute_missing_values(df, impute):
    missed_columns = df.columns[df.isna().any()].tolist()
    imp_values = {}
    for col in missed_columns:
        if df[col].dtypes in ["object", "bool"]: # Object
            value = df[col].mode()
        else: # Numeric
            num_missed = impute.split("/")[0]
            if num_missed=="Mean":
                value = df[col].mean()
            else:
                value = df[col].median()
        df[col] = df[col].fillna(value)
        imp_values[col] = value

    return df, imp_values

@st.cache_data
def sampling_data(full_df, sample_size, _id, group=None):
    sample_ratio = sample_size / full_df.shape[0]
    if group:
        sample_df = pd.DataFrame()
        if full_df[group].dtypes in ["int64", "float64"]:
            full_df["binned"] = pd.qcut(full_df[group], q=10)
            group = "binned"

        for group_name in full_df[group].unique():
            group_full_df = full_df[full_df[group]==group_name].reset_index(drop=True)
            group_sample_df = group_full_df.sample(
                frac=sample_ratio, random_state=42, ignore_index=True
            )
            sample_df = pd.concat([sample_df, group_sample_df])

        if group=="binned":
            sample_df = sample_df.drop("binned", axis=1)

    else:
        sample_df = full_df.sample(frac=sample_ratio, random_state=42, ignore_index=True)
    sample_df = sample_df.sort_values(by=_id, ignore_index=True)

    return sample_df
