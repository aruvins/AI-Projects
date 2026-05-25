import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate(model, X_test, y_test, name="model"):
    os.makedirs("outputs/plots", exist_ok=True)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)

    print(f"\n📊 {name}")
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)

    # -----------------------------
    # 1. Actual vs Predicted Plot
    # -----------------------------
    plt.figure()

    plt.scatter(y_test, preds)
    plt.xlabel("Actual House Price")
    plt.ylabel("Predicted House Price")
    plt.title(f"{name} — Actual vs Predicted")

    # Perfect prediction line
    min_val = min(min(y_test), min(preds))
    max_val = max(max(y_test), max(preds))

    plt.plot([min_val, max_val], [min_val, max_val])

    plt.savefig(f"outputs/plots/{name}_actual_vs_pred.png")
    plt.show()

    # -----------------------------
    # 2. Residual Distribution
    # -----------------------------
    residuals = y_test - preds

    plt.figure()

    plt.hist(residuals, bins=40)
    plt.xlabel("Error (Residual)")
    plt.ylabel("Frequency")
    plt.title(f"{name} — Residual Distribution")

    plt.savefig(f"outputs/plots/{name}_residuals.png")
    plt.show()

    # -----------------------------
    # 3. Residual vs Predicted
    # -----------------------------
    plt.figure()

    plt.scatter(preds, residuals)
    plt.axhline(y=0, color="r", linestyle="--")

    plt.xlabel("Predicted Value")
    plt.ylabel("Residual Error")
    plt.title(f"{name} — Residuals vs Predictions")

    plt.savefig(f"outputs/plots/{name}_residual_vs_pred.png")
    plt.show()