"""
Uses the same preprocessing utilities as train.py and keeps outputs consistent.
"""
from sklearn.kernel_ridge import KernelRidge
from misc import load_data, split_and_scale, fit_model, compute_mse, save_artifact
import os

def main():
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df, test_size=0.2, random_state=42, scale=True)

    model = KernelRidge(alpha=1.0, kernel='rbf')

    model = fit_model(model, X_train, y_train)

    mse = compute_mse(model, X_test, y_test)
    print(f"KernelRidge Test MSE: {mse:.4f}")

    os.makedirs("artifacts", exist_ok=True)
    save_artifact(model, "artifacts/kernelridge_model.joblib")
    if scaler is not None:
        save_artifact(scaler, "artifacts/scaler.joblib")

if __name__ == "__main__":
    main()

