"""
utils.py — Utility functions for IBM HR Analytics Project
Fully robust EDA and preprocessing functions
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.patches as mpatches  # Correct patch type for bars

def quick_overview(df):
    """Print first 5 rows, info, and basic stats"""
    print("=== Head of dataset ===")
    print(df.head(), "\n")
    print("=== Info ===")
    print(df.info(), "\n")
    print("=== Describe ===")
    print(df.describe(), "\n")

def numeric_summary(df):
    """Print summary statistics for numeric columns"""
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    if numeric_df.empty:
        print("⚠️ No numeric columns found in dataset!")
    else:
        print("=== Numeric Summary ===")
        print(numeric_df.describe())

def visualize_attrition(df, col='Attrition', figures_dir=None, show_plot=True):
    """Count plot of Attrition with percentages and optional save"""
    if col not in df.columns:
        print(f"⚠️ Column '{col}' not found in DataFrame!")
        return

    # Convert column to string in case it's boolean or numeric
    df_plot = df[col].astype(str)

    plt.figure(figsize=(6,4))
    total = len(df_plot)
    ax = sns.countplot(x=df_plot)

    # Add percentages safely
    for patch in ax.patches:
        if isinstance(patch, mpatches.Rectangle):
            try:
                height = patch.get_height()
                percent = f'{100 * height / total:.1f}%'
                ax.text(patch.get_x() + patch.get_width()/2., height + 0.5, percent,
                        ha="center", fontsize=11)
            except Exception as e:
                print("⚠️ Warning while adding percentages:", e)

    plt.title(f"{col} Count with Percentages")
    plt.tight_layout()

    if figures_dir:
        os.makedirs(figures_dir, exist_ok=True)
        save_path = os.path.join(figures_dir, f"{col}_count_plot.png")
        plt.savefig(save_path)
        print(f"✅ Plot saved to {save_path}")

    if show_plot:
        plt.show()
    plt.close()

def correlation_heatmap(df, figures_dir=None, show_plot=True):
    """Heatmap of correlations (numeric columns only, NaNs filled)"""
    df_copy = df.copy()
    # Convert boolean columns to int
    for col in df_copy.select_dtypes(include=['bool']).columns:
        df_copy[col] = df_copy[col].astype(int)

    numeric_df = df_copy.select_dtypes(include=['int64', 'float64']).fillna(0)
    if numeric_df.empty or numeric_df.nunique().sum() == 0:
        print("⚠️ No numeric columns with variance found for correlation heatmap!")
        return

    plt.figure(figsize=(12,10))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()

    if figures_dir:
        os.makedirs(figures_dir, exist_ok=True)
        save_path = os.path.join(figures_dir, "correlation_heatmap.png")
        plt.savefig(save_path)
        print(f"✅ Heatmap saved to {save_path}")

    if show_plot:
        plt.show()
    plt.close()
