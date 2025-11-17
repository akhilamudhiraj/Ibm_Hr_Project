"""
main.py — IBM HR Analytics Project
Fully robust script for EDA, modeling, hyperparameter tuning,
feature importance visualization, and saving outputs.
"""

import os
import pandas as pd
import pickle
from utils import quick_overview, numeric_summary, visualize_attrition, correlation_heatmap

# ML imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------- Output folders ----------------------
FIGURES_DIR = "figures"
MODELS_DIR = "models"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------- Utility Functions ----------------------
def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    return rf

def tune_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, scoring='accuracy')
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------------------- Main Function ----------------------
def main():
    # Load dataset
    data_path = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    if not os.path.exists(data_path):
        print(f"⚠️ Dataset not found at: {data_path}")
        return

    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()  # Clean column names
    print("✅ Dataset loaded successfully!\n")

    # ---------------------- EDA ----------------------
    quick_overview(df)
    numeric_summary(df)
    visualize_attrition(df, col='Attrition', figures_dir=FIGURES_DIR)
    correlation_heatmap(df, figures_dir=FIGURES_DIR)

    # ---------------------- Prepare Data ----------------------
    if 'Attrition' not in df.columns:
        print("⚠️ 'Attrition' column not found, cannot train model.")
        return

    df_encoded = pd.get_dummies(df, drop_first=True)
    target_col = 'Attrition_Yes'
    if target_col not in df_encoded.columns:
        print(f"⚠️ Target column '{target_col}' not found after encoding.")
        return

    X = df_encoded.drop(target_col, axis=1)
    y = df_encoded[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------------------- Train Initial Random Forest ----------------------
    print("\n--- Training Initial Random Forest ---")
    rf_model = train_random_forest(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test)

    # ---------------------- Hyperparameter Tuning ----------------------
    print("\n--- Hyperparameter Tuning ---")
    best_model, best_params = tune_random_forest(X_train, y_train)
    print("Best Parameters:", best_params)
    evaluate_model(best_model, X_test, y_test)

    # Save trained model
    model_path = os.path.join(MODELS_DIR, "rf_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"✅ Trained model saved to {model_path}")

    # ---------------------- Feature Importance ----------------------
    feature_importances = best_model.feature_importances_
    features = X_train.columns

    feat_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    feat_df = feat_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feat_df)
    plt.title("Feature Importance - Random Forest")
    plt.tight_layout()
    feat_path = os.path.join(FIGURES_DIR, "feature_importance.png")
    plt.savefig(feat_path)
    plt.show()
    plt.close()
    print(f"✅ Feature importance plot saved to {feat_path}")

# ---------------------- Run Main ----------------------
if __name__ == "__main__":
    main()
