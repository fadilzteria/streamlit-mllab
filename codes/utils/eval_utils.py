import random
import copy
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix
import streamlit as st

from codes.utils import classification as classif, regression as regress, test_utils

COLORS = list(mcolors.XKCD_COLORS.keys())
random.Random(1).shuffle(COLORS)

@st.cache_data
def show_metrics(config, metric_df, fold="All"):
    # Filter Dataframe
    if fold=="All": # Average All Folds
        df = metric_df.groupby("Model").mean().reset_index()
    else:
        df = metric_df[metric_df["Fold"]==fold].reset_index(drop=True)
    df = df.drop("Fold", axis=1)
    if config["best_value"]=="Maximize":
        df = df.sort_values(
            by=f"Valid {config['best_metric']}", ascending=False
        ).reset_index(drop=True)
    else:
        df = df.sort_values(
            by=f"Valid {config['best_metric']}", ascending=True
        ).reset_index(drop=True)

    x_axis = np.arange(len(df))
    model_names = df["Model"].tolist()
    wrapper = textwrap.TextWrapper(width=25)
    model_names = [
        "\n".join(wrapper.wrap(text=model_name)) for model_name in model_names
    ]

    metric_list = config["metrics"]

    for metric_name in metric_list:
        # Create Subplots
        fig, ax = plt.subplots(figsize=(15, 5))

        # Multiple Vertical Bar Chart
        train_metrics = df[f"Train {metric_name}"].tolist()
        valid_metrics = df[f"Valid {metric_name}"].tolist()
        full_metrics = train_metrics + valid_metrics

        plt.bar(x_axis-0.15, train_metrics, 0.3, label='Train', color=COLORS[0])
        plt.bar(x_axis+0.15, valid_metrics, 0.3, label='Valid', color=COLORS[1])

        # Text
        max_ratio = 105
        min_value = min(full_metrics)
        min_value = min(min_value, 0)
        max_value = max(full_metrics)
        add_text_value = (max_value-min_value)*(max_ratio-100)/100

        for k, (train_v, valid_v) in enumerate(zip(train_metrics, valid_metrics)):
            ax.text(
                k-0.15, train_v+add_text_value, str(round(train_v, 2)),
                ha='center', va='center', fontsize=10
            )
            ax.text(
                k+0.15, valid_v+add_text_value, str(round(valid_v, 2)),
                ha='center', va='center', fontsize=10
            )

        # Update Axis
        plt.xticks(x_axis, model_names, fontsize=12)
        sns.despine(top=True, right=True)

        # Show
        plt.title(f"{metric_name} Results for Fold {fold}", size=14, y=1.15)
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig)

@st.cache_data
def get_class_metrics(config, metric_df, fold="All"):
    # Filter Dataframe
    if fold=="All": # Average All Folds
        df = metric_df.groupby("Model").mean().reset_index()
    else:
        df = metric_df[metric_df["Fold"]==fold].reset_index(drop=True)

    metric_list = config["metrics"]
    if "Accuracy" in metric_list:
        metric_list.remove("Accuracy")
    nrows = len(metric_list)

    fig, axs = plt.subplots(nrows=nrows, ncols=2, figsize=(15, nrows*4))
    wrapper = textwrap.TextWrapper(width=12)

    for j, split in enumerate(["Train", "Valid"]):
        for i, metric in enumerate(metric_list):
            # Class Metric Dataframe
            class_metric_columns = ["Model"] + \
                df.filter(regex=f"{split} {metric}-").columns.tolist()
            class_metric_df = df[class_metric_columns]
            class_metric_df = class_metric_df.set_index("Model")
            class_metric_df.index.name = None
            class_metric_df.columns = [col.split("-")[-1] for col in class_metric_df.columns]
            class_metric_df = class_metric_df.transpose()
            class_metric_df.columns = [
                "\n".join(wrapper.wrap(text=col)) for col in class_metric_df.columns
            ]

            # Heatmap
            ax_temp = axs[i] if nrows==1 else axs[i, j]
            _ = sns.heatmap(
                class_metric_df, annot=class_metric_df, fmt=".3f", annot_kws={"fontsize": 9},
                ax=ax_temp, cmap='RdYlGn', vmin=0.0, vmax=1.0, linewidths=0.5, linecolor='White',
                square=True, cbar=False
            )

            # Update Axes
            ax_temp.set_title(f"{split} {metric}", size=12)
            ax_temp.set_xticklabels(ax_temp.get_xticklabels(), rotation=0, fontsize=10)
            ax_temp.set_yticklabels(ax_temp.get_yticklabels(), rotation=0, fontsize=10)

    plt.tight_layout()
    plt.suptitle(f"Class-wise Metrics for Fold {fold}", y=1.02, size=15)
    st.pyplot(fig)

@st.cache_data
def show_confusion_matrix(config, oof_df, fold="All"):
    # Filter Dataframe
    if fold!="All": # Average All Folds
        oof_df = oof_df[oof_df["fold"]==fold].reset_index(drop=True)

    # Parameters
    y_test = oof_df[config["target"]]
    class_names = oof_df[config["target"]].unique().tolist()
    model_names = test_utils.extract_model_names(config)

    # Create Subplots
    ncols = 3
    nrows = ((len(model_names)-1)//ncols)+1
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows*5))

    for i, model_name in enumerate(model_names):
        y_pred = oof_df[f"{model_name}_{config['target']}_pred"]
        cn_matrix = confusion_matrix(y_test, y_pred, labels=class_names)
        cn_matrix = pd.DataFrame(cn_matrix)
        cn_matrix.index, cn_matrix.columns = class_names, class_names

        # Heatmap
        ax_temp = axs[i] if nrows==1 else axs[i//ncols, i%ncols]
        _ = sns.heatmap(
            cn_matrix, annot=cn_matrix, fmt="", annot_kws={"fontsize": 8}, ax=ax_temp,
            cmap='Greens', linewidths=0.5, linecolor='White', square=True, cbar=False,
        )
        ax_temp.set_title(model_name, size=10)
        ax_temp.set(xlabel="Predicted", ylabel="Actual")

    if (len(model_names)-1) < (nrows*ncols):
        for j in range(len(model_names), (nrows*ncols)):
            ax_temp = axs[j] if nrows==1 else axs[j//ncols, j%ncols]
            ax_temp.axis("off")

    plt.suptitle(f"Confusion Matrix for Fold {fold}", y=0.92)
    st.pyplot(fig)

@st.cache_data
def get_category_metrics(config, cleaned_df, oof_df, fold="All", max_unique=4, numeric=False):
    # Filter Dataframe
    if fold!="All": # Average All Folds
        oof_df = oof_df[oof_df["fold"]==fold]
        oof_df_idxs = oof_df.index.tolist()
        cleaned_df = cleaned_df[cleaned_df.index.isin(oof_df_idxs)]

    cat_df_dict = {}
    for df_col in cleaned_df.columns:
        if config["target"]==df_col:
            continue

        if cleaned_df[df_col].dtypes in ["object", "bool"]:
            if cleaned_df[df_col].nunique() > max_unique or "pred" in df_col:
                continue
        else:
            if numeric and "pred" not in df_col:
                cleaned_df[f"{df_col}_binned"] = pd.qcut(
                    cleaned_df[df_col], q=max_unique, duplicates='drop'
                ).astype('str')
                df_col = f"{df_col}_binned"
            else:
                continue

        cat_df_dict[df_col] = []
        metric_list = config["metrics"]

        oof_df["group"] = copy.deepcopy(cleaned_df[df_col])

        # For Each Category in the Categorical Column
        for _, category in enumerate(oof_df["group"].unique()):
            cat_df = oof_df[oof_df["group"]==category].reset_index(drop=True)
            cat_metric_df = pd.DataFrame()
            # For Each Model
            model_names = [col.split("_")[0] for col in cat_df.columns if "pred" in col]
            for model_name in model_names:
                # Metrics
                y_true = cat_df[config["target"]]
                y_pred = cat_df[f"{model_name}_{config['target']}_pred"]
                model_metric_df = pd.DataFrame({"Model": model_name}, index=[0])

                if config["ml_task"]=="Classification":
                    model_proba_names = [
                        col for col in cat_df.columns if model_name in col and "proba" in col
                    ]
                    y_pred_proba = np.array(cat_df[model_proba_names])

                    if cat_df[config["target"]].nunique() == 2:
                        model_metric_df = classif.get_binary_results(
                            y_true, y_pred, y_pred_proba, model_metric_df, metric_list, split=""
                        )
                    else:
                        class_names = cat_df[config["target"]].unique().tolist()
                        model_metric_df = classif.get_multi_results(
                            y_true, y_pred, y_pred_proba, model_metric_df, metric_list, split="",
                            class_names=class_names
                        )
                else:
                    n, p = cat_df.shape[0], cat_df.shape[1]
                    model_metric_df = regress.get_results(
                        y_true, y_pred, model_metric_df, metric_list, split="", n=n, p=p
                    )

                cat_metric_df = pd.concat([cat_metric_df, model_metric_df], ignore_index=True)

            cat_df_dict[df_col].append({"category": category, "data": cat_metric_df})

    return cat_df_dict

@st.cache_data
def show_category_metrics(cat_df_dict, group_col):
    group_cat_df_list = cat_df_dict[group_col]
    for group_cat_df in group_cat_df_list:
        category = group_cat_df["category"]
        cat_metric_df = group_cat_df["data"]
        st.write(category)
        st.dataframe(cat_metric_df)

@st.cache_data
def show_runtime(metric_df, fold="All"):
    # Filter Dataframe
    if fold=="All": # Average All Folds
        df = metric_df.groupby("Model").mean().reset_index()
    else:
        df = metric_df[metric_df["Fold"]==fold].reset_index(drop=True)
    df = df.drop("Fold", axis=1)
    df = df.sort_values(by="Training Runtime", ascending=False).reset_index(drop=True)

    x_axis = np.arange(len(df))
    model_names = df["Model"].tolist()
    wrapper = textwrap.TextWrapper(width=25)
    model_names = [
        "\n".join(wrapper.wrap(text=model_name)) for model_name in model_names
    ]

    # Create Subplots
    fig, ax = plt.subplots(figsize=(15, 5.5))

    # Multiple Vertical Bar Chart
    train_runtime = df["Training Runtime"].tolist()
    pred_runtime = df["Prediction Runtime"].tolist()
    full_runtime = train_runtime + pred_runtime

    plt.bar(x_axis-0.15, train_runtime, 0.3, label='Training', color=COLORS[0])
    plt.bar(x_axis+0.15, pred_runtime, 0.3, label='Prediction', color=COLORS[1])

    # Text
    max_ratio = 105
    min_value = min(full_runtime)
    max_value = max(full_runtime)
    add_text_value = (max_value-min_value)*(max_ratio-100)/100

    for k, (train_v, valid_v) in enumerate(zip(train_runtime, pred_runtime)):
        ax.text(
            k-0.15, train_v+add_text_value, str(round(train_v, 2)),
            ha='center', va='center', fontsize=10
        )
        ax.text(
            k+0.15, valid_v+add_text_value, str(round(valid_v, 2)),
            ha='center', va='center', fontsize=10
        )

    # Update Axis
    plt.xticks(x_axis, model_names, fontsize=12)
    sns.despine(top=True, right=True)

    # Show
    plt.title(f"Runtime Results for Fold {fold}", size=14, y=1.15)
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)

@st.cache_data
def show_reg_diagnostics(config, oof_df, fold="All"):
    # Filter Dataframe
    if fold!="All": # Average All Folds
        oof_df = oof_df[oof_df["fold"]==fold].reset_index(drop=True)

    # Parameters
    model_names = test_utils.extract_model_names(config)

    # Create Subplots
    ncols = 3
    nrows = len(model_names)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, nrows*4))
    wrapper = textwrap.TextWrapper(width=35)

    for i, model_name in enumerate(model_names):
        actual = oof_df[config['target']] # Actual
        pred = oof_df[f"{model_name}_{config['target']}_pred"] # Prediction
        res = actual - pred # Residuals
        std_res = (res - res.mean())/res.std() # Standardized Residuals

        # Linearity
        ax_temp = axs[0] if nrows==1 else axs[i, 0]
        ax_temp.scatter(actual, pred, c=COLORS[0], s=8, alpha=0.8)
        title = f"{model_name} - Linearity"
        title = "\n".join(wrapper.wrap(text=title))
        ax_temp.set_title(title, size=11)
        ax_temp.set(xlabel="Actual", ylabel="Predicted")

        # Residuals vs Predicted
        ax_temp = axs[1] if nrows==1 else axs[i, 1]
        ax_temp.scatter(pred, res, c=COLORS[1], s=8, alpha=0.8)
        title = f"{model_name} - Residuals vs Predicted"
        title = "\n".join(wrapper.wrap(text=title))
        ax_temp.set_title(title, size=11)
        ax_temp.set(xlabel="Predicted", ylabel="Residuals")

        # Normal Q-Q Plot
        ax_temp = axs[2] if nrows==1 else axs[i, 2]
        _ = sm.qqplot(
            std_res, line='45',
            marker='o', markerfacecolor=COLORS[0], markeredgecolor=COLORS[2],
            alpha=0.8, ax=ax_temp
        )
        title = f"{model_name} - Normal Q-Q"
        title = "\n".join(wrapper.wrap(text=title))
        ax_temp.set_title(title, size=11)

    # Show
    plt.suptitle(f"Regression Diagnostics for Fold {fold}", y=1.0, size=15)
    plt.tight_layout()
    st.pyplot(fig)

@st.cache_data
def show_regress_predicted_distribution(config, oof_df, fold="All"):
    # Filter Dataframe
    if fold!="All": # Average All Folds
        oof_df = oof_df[oof_df["fold"]==fold].reset_index(drop=True)

    # Parameters
    model_names = test_utils.extract_model_names(config)

    # Create Subplots
    ncols = 3
    nrows = ((len(model_names)-1)//ncols)+1
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows*5))
    wrapper = textwrap.TextWrapper(width=40)

    for i, model_name in enumerate(model_names):
        ax_temp = axs[i] if nrows==1 else axs[i//ncols, i%ncols]
        pred_name = f"{model_name}_{config['target']}_pred"

        sns.kdeplot(
            oof_df[config['target']], color=COLORS[0], fill=True, label="Actual", ax=ax_temp
        )
        sns.kdeplot(oof_df[pred_name], color=COLORS[1], fill=True, label="Predicted", ax=ax_temp)

        model_name = "\n".join(wrapper.wrap(text=model_name))
        ax_temp.set_title(model_name, size=11)
        ax_temp.set(xlabel=None, ylabel=None)
    ax_temp = axs[len(model_names)-1] if nrows==1 else axs[0, 2]
    ax_temp.legend()

    if (len(model_names)-1) < (nrows*ncols):
        for j in range(len(model_names), (nrows*ncols)):
            ax_temp = axs[j] if nrows==1 else axs[j//ncols, j%ncols]
            ax_temp.axis("off")

    plt.suptitle(f"Predicted Distribution Plot for Fold {fold}", y=1.0, size=15)
    plt.tight_layout()
    st.pyplot(fig)

@st.cache_data
def show_classif_predicted_distribution(config, oof_df, fold="All"):
    # Filter Dataframe
    if fold!="All": # Average All Folds
        oof_df = oof_df[oof_df["fold"]==fold].reset_index(drop=True)

    # Parameters
    model_names = test_utils.extract_model_names(config)
    target_names = oof_df[config['target']].unique().tolist()

    # Create Subplots
    ncols = len(target_names)
    nrows = len(model_names)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*3, nrows*3))
    wrapper = textwrap.TextWrapper(width=35)

    for i, model_name in enumerate(model_names):
        for j, target_name in enumerate(target_names):
            ax_temp = axs[j] if nrows==1 else axs[i, j]
            pred_name = f"{model_name}_{config['target']}_{target_name}_proba"
            target_df = oof_df[oof_df[config['target']]==target_name].reset_index(drop=True)

            sns.histplot(target_df[pred_name], color=COLORS[j], bins=20, ax=ax_temp)

            title = f"{model_name} for {target_name}"
            title = "\n".join(wrapper.wrap(text=title))
            ax_temp.set_title(title, size=9)
            ax_temp.set(xlabel=None, ylabel=None)

    plt.suptitle(f"Predicted Distribution Plot for Fold {fold}", y=1.0, size=12)
    plt.tight_layout()
    st.pyplot(fig)
