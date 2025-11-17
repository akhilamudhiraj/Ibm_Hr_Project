"""
model.py — Model training, hyperparameter tuning, and evaluation
for IBM HR Analytics Project
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_random_forest(X_train, y_train):
    """
    Train a basic Random Forest model
    """
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    return rf

def tune_random_forest(X_train, y_train):
    """
    Hyperparameter tuning using GridSearchCV
    """
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    return best_model, best_params

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance and print metrics
    """
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    return y_pred
