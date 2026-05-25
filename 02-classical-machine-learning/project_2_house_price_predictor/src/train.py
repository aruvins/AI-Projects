from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

import joblib
import os

def train_models(preprocessor, X_train, y_train):
    os.makedirs("outputs/model", exist_ok=True)

    # Linear Regression
    lr_model = Pipeline([
        ("preprocess", preprocessor),
        ("model", LinearRegression())
    ])

    lr_model.fit(X_train, y_train)
    joblib.dump(lr_model, "outputs/model/linear_regression.pkl")

    # Random Forest
    rf_model = Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, "outputs/model/random_forest.pkl")

    print("Training complete")

    return lr_model, rf_model