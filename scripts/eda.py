"""
eda.py — Exploratory Data Analysis for IBM HR Analytics Project
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def quick_overview(df: pd.DataFrame):
    """Show basic info and attrition counts."""
    print("=== Dataset Overview ===")
    print(df.info())
    print("\n=== Missing Values per Column ===")
    print(df.isnull().sum())
    print("\n=== Attrition Distribution ===")
    print(df["Attrition"].value_counts())
    print("\n")

def numeric_summary(df: pd.DataFrame):
    """Show statistical summary of numerical columns."""
    print("=== Numerical Summary ===")
    print(df.describe().T)
    print("\n")

def visualize_attrition(df: pd.DataFrame):
    """Visualize attrition count and department-wise attrition."""
    plt.figure(figsize=(6, 4))
    sns.countplot(x="Attrition", data=df, palette="Set2")
    plt.title("Overall Attrition Count")
    plt.show()

    if "Department" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(x="Department", hue="Attrition", data=df, palette="pastel")
        plt.title("Attrition by Department")
        plt.xticks(rotation=15)
        plt.show()

def correlation_heatmap(df: pd.DataFrame):
    """Show correlation heatmap for numerical columns."""
    plt.figure(figsize=(10, 6))
    corr = df.select_dtypes(include="number").corr()
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    plt.title("Correlation Heatmap")
    plt.show()
