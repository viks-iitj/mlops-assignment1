"""
This script uses utility functions from misc.py and saves the trained model and scaler.
"""
from sklearn.tree import DecisionTreeRegressor
from misc import load_data, split_and_scale, fit_model, compute_mse, save_artifact
import os

def main():
    # Load data
    df = load_data()

    # Prepare data (split + scale)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df, test_size=0.2, random_state=42, scale=True)

    # Initialize model
    model = DecisionTreeRegressor(random_state=42, max_depth=6)

    # Train
    model = fit_model(model, X_train, y_train)

    # Evaluate
    mse = compute_mse(model, X_test, y_test)
    print(f"DecisionTreeRegressor Test MSE: {mse:.4f}")

    # Save artifacts
    os.makedirs("artifacts", exist_ok=True)
    save_artifact(model, "artifacts/dtree_model.joblib")
    if scaler is not None:
        save_artifact(scaler, "artifacts/scaler.joblib")

if __name__ == "__main__":
    main()

