import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from sklearn.metrics import confusion_matrix

import streamlit as st

COLORS = list(mcolors.XKCD_COLORS.keys())
random.Random(1).shuffle(COLORS)

def show_metrics(config, metric_df, fold="All"):
    # Filter Dataframe
    if(fold=="All"): # Average All Folds
        df = metric_df.groupby("Model").mean().reset_index()
    else:
        df = metric_df[metric_df["Fold"]==fold].reset_index(drop=True)
    df = df.drop("Fold", axis=1)
    if(config["best_value"]=="Maximize"):
        df = df.sort_values(by=f"Valid {config['best_metric']}", ascending=False).reset_index(drop=True)
    else:
        df = df.sort_values(by=f"Valid {config['best_metric']}", ascending=True).reset_index(drop=True)

    x_axis = np.arange(len(df))
    model_names = df["Model"].tolist()

    metric_list = config["metrics"]

    for metric_name in metric_list:
        # Create Subplots
        fig, _ = plt.subplots(figsize=(15, 4))

        # Multiple Vertical Bar Chart
        train_metrics = df[f"Train {metric_name}"].tolist()
        valid_metrics = df[f"Valid {metric_name}"].tolist()

        plt.bar(x_axis-0.15, train_metrics, 0.3, label='Train', color=COLORS[0])
        plt.bar(x_axis+0.15, valid_metrics, 0.3, label='Valid', color=COLORS[1])

        # Update Axis
        plt.xticks(x_axis, model_names)
        sns.despine(top=True, right=True)

        # Show
        plt.title(f"{metric_name} Results for Fold {fold}")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig)

def get_class_metrics(config, metric_df, fold="All"):
    # Filter Dataframe
    if(fold=="All"): # Average All Folds
        df = metric_df.groupby("Model").mean().reset_index()
    else:
        df = metric_df[metric_df["Fold"]==fold].reset_index(drop=True)

    fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(15, 24))
    metric_list = config["metrics"]
    if("Accuracy" in metric_list):
        metric_list.remove("Accuracy")

    for j, split in enumerate(["Train", "Valid"]):
        for i, metric in enumerate(metric_list):
            # Class Metric Dataframe
            class_metric_columns = ["Model"] + df.filter(regex=f"{split} {metric}-").columns.tolist()
            class_metric_df = df[class_metric_columns]
            class_metric_df = class_metric_df.set_index("Model")
            class_metric_df.index.name = None
            class_metric_df.columns = [col.split("-")[-1] for col in class_metric_df.columns]
            class_metric_df = class_metric_df.transpose()

            # Heatmap
            im = sns.heatmap(
                class_metric_df, annot=class_metric_df, fmt=".3f", annot_kws={"fontsize": 8}, ax=axs[i, j],
                cmap='RdYlGn', vmin=0.0, vmax=1.0, linewidths=0.5, linecolor='White', square=True, cbar=False
            )

            # Update Axes
            axs[i, j].set_title(f"{split} {metric} for each Classes", size=9)

    plt.suptitle(f"Class-wise Metrics for Fold {fold}", y=0.90)
    st.pyplot(fig)

def show_confusion_matrix(config, df):
    y_test = df[config["target"]]
    class_names = df[config["target"]].unique().tolist()

    # Create Subplots
    ncols = 3
    nrows = ((len(config["methods"])-1)//ncols)+1
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows*5))

    for i, model_name in enumerate(config["methods"]):
        y_pred = df[f"{model_name}_{config['target']}_pred"]
        cn_matrix = confusion_matrix(y_test, y_pred, labels=class_names)
        cn_matrix = pd.DataFrame(cn_matrix)
        cn_matrix.index, cn_matrix.columns = class_names, class_names

        # Heatmap
        ax_temp = axs[i] if nrows==1 else axs[i//ncols, i%ncols]
        im = sns.heatmap(
            cn_matrix, annot=cn_matrix, fmt="", annot_kws={"fontsize": 8}, ax=ax_temp,
            cmap='Greens', linewidths=0.5, linecolor='White', square=True, cbar=False,
        )
        ax_temp.set_title(model_name, size=10)
        ax_temp.set(xlabel="Predicted", ylabel="Actual")

    if(i < (nrows*ncols)):
        for j in range(i+1, (nrows*ncols)):
            ax_temp = axs[j] if nrows==1 else axs[j//ncols, j%ncols]
            ax_temp.axis("off")

    plt.suptitle("Confusion Matrix", y=0.92)
    st.pyplot(fig)

def show_runtime(metric_df, fold="All"):
    # Filter Dataframe
    if(fold=="All"): # Average All Folds
        df = metric_df.groupby("Model").mean().reset_index()
    else:
        df = metric_df[metric_df["Fold"]==fold].reset_index(drop=True)
    df = df.drop("Fold", axis=1)
    df = df.sort_values(by="Training Runtime", ascending=False).reset_index(drop=True)

    x_axis = np.arange(len(df))
    model_names = df["Model"].tolist()

    # Create Subplots
    fig, _ = plt.subplots(figsize=(15, 5))

    # Multiple Vertical Bar Chart
    train_runtime = df["Training Runtime"].tolist()
    pred_runtime = df["Prediction Runtime"].tolist()

    plt.bar(x_axis-0.15, train_runtime, 0.3, label='Training', color=COLORS[0])
    plt.bar(x_axis+0.15, pred_runtime, 0.3, label='Prediction', color=COLORS[1])

    # Update Axis
    plt.xticks(x_axis, model_names)
    sns.despine(top=True, right=True)

    # Show
    plt.title(f"Runtime Results for Fold {fold}")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)