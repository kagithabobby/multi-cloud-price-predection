from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from ml.config import FEATURE_COLUMNS, TARGET_COLUMN, ProjectPaths
from ml.preprocess import build_preprocessor


def _assert_schema(df: pd.DataFrame) -> None:
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(missing)
            + ". Please check your CSV header."
        )


def _print_interpretation_help(feature_names: list[str]) -> None:
    print("\n--- Coefficient interpretation (how to read the model) ---")
    print(
        "Linear Regression predicts:\n"
        "  y = intercept + sum_i( coef_i * feature_i )\n"
        "Here, numeric features were standardized (z-score). That means:\n"
        "- A coefficient tells you how much the monthly cost changes when that\n"
        "  feature increases by 1 standard deviation, holding others constant.\n"
        "- Provider dummy coefficients are differences vs AWS (baseline)."
    )
    print("\nFeatures the model sees (after preprocessing):")
    for name in feature_names:
        print(f"  - {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train linear regression cost model.")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to the CSV dataset. Default: data/cloud_costs.csv",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction for test split (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for split (default: 42).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    paths = ProjectPaths(project_root=project_root)

    csv_path = Path(args.csv) if args.csv else paths.default_csv_path
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found at: {csv_path}\n"
            "Put your dataset in data/ (e.g. data/cloud_costs.csv) "
            "or pass --csv path/to/your.csv"
        )

    df = pd.read_csv(csv_path)
    _assert_schema(df)

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(float).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    # WHAT: Fit preprocessing ONLY on training data.
    # WHY: Prevents data leakage (test set shouldn't influence scaling stats).
    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_t, y_train)

    preds = model.predict(X_test_t)
    mae = mean_absolute_error(y_test, preds)
    # Older versions of scikit-learn don't support the `squared` argument.
    # To stay version-agnostic, we compute RMSE manually.
    mse = mean_squared_error(y_test, preds)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, preds)

    print("\n--- Evaluation on held-out test set ---")
    print(f"MAE : {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R^2 : {r2:,.4f}")

    feature_names = list(preprocessor.get_feature_names_out())
    _print_interpretation_help(feature_names)

    coef_pairs = sorted(zip(feature_names, model.coef_), key=lambda t: abs(t[1]), reverse=True)
    print("\nTop coefficients by absolute magnitude:")
    for name, coef in coef_pairs[:10]:
        print(f"  {name:>24s} : {coef: .4f}")

    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, paths.artifacts_dir / "model.pkl")

    # NOTE: Requirement asks for scaler.pkl. In production we'd call this
    # preprocessor.pkl because it includes BOTH scaling and one-hot encoding.
    # We keep the requested filename for clarity in your learning path.
    joblib.dump(preprocessor, paths.artifacts_dir / "scaler.pkl")

    meta = {
        "feature_columns_raw": FEATURE_COLUMNS,
        "feature_columns_model": feature_names,
        "intercept": float(model.intercept_),
        "coefficients": {name: float(coef) for name, coef in zip(feature_names, model.coef_)},
        "metrics": {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)},
    }
    joblib.dump(meta, paths.artifacts_dir / "training_meta.pkl")

    print("\nSaved artifacts to artifacts/:")
    print("  - artifacts/model.pkl")
    print("  - artifacts/scaler.pkl")
    print("  - artifacts/training_meta.pkl")


if __name__ == "__main__":
    main()

