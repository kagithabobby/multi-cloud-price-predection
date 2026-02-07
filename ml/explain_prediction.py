"""
Step-by-step: how one prediction is computed from raw inputs.

Run from project root:
  python -m ml.explain_prediction

Uses one row from your dataset and the saved model/scaler to show:
  1. Raw input values
  2. One-hot encoding (AWS as baseline)
  3. Z-score standardization (with actual mean/std from training)
  4. The linear equation: predicted_cost = intercept + sum(coef_i * feature_i)
  5. Verification against model.predict()
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.config import CLOUD_PROVIDER_CATEGORIES, TARGET_COLUMN
from ml.preprocess import NUMERIC_FEATURES


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    artifacts_dir = project_root / "artifacts"
    csv_path = project_root / "data" / "multicloud_cost_prediction_dataset.csv"

    model = joblib.load(artifacts_dir / "model.pkl")
    preprocessor = joblib.load(artifacts_dir / "scaler.pkl")

    # Pick one row from your dataset (first row)
    df = pd.read_csv(csv_path)
    row = df.iloc[0].to_dict()
    actual_cost = row[TARGET_COLUMN]

    print("=" * 60)
    print("STEP-BY-STEP: How the model computes one prediction")
    print("=" * 60)

    print("\n--- 1) RAW INPUT (one row from your CSV) ---")
    for k, v in row.items():
        if k == TARGET_COLUMN:
            print(f"  {k} (target, not used as input): {v}")
        else:
            print(f"  {k}: {v}")

    provider = row["cloud_provider"]

    print("\n--- 2) ONE-HOT ENCODING (cloud_provider, AWS = baseline) ---")
    print("  Categories (fixed order):", CLOUD_PROVIDER_CATEGORIES)
    print("  We DROP the first category (AWS) so it becomes the reference.")
    print("  So we create two columns: Azure_dummy, GCP_dummy")
    print()
    azure_dummy = 1.0 if provider == "Azure" else 0.0
    gcp_dummy = 1.0 if provider == "GCP" else 0.0
    print(f"  For provider = '{provider}':")
    print(f"    cloud_provider_Azure = {azure_dummy:.0f}")
    print(f"    cloud_provider_GCP   = {gcp_dummy:.0f}")
    print("  (Interpretation: AWS = 0,0 = baseline; Azure = 1,0; GCP = 0,1)")

    print("\n--- 3) Z-SCORE STANDARDIZATION (numeric features) ---")
    print("  Formula: z = (x - mean) / std")
    print("  Mean and std were computed on the TRAINING set only.\n")

    scaler = preprocessor.named_transformers_["num"]
    means = scaler.mean_
    stds = scaler.scale_  # StandardScaler stores std in .scale_

    z_values = []
    for i, name in enumerate(NUMERIC_FEATURES):
        x = float(row[name])
        mu = means[i]
        sigma = stds[i]
        z = (x - mu) / sigma if sigma > 0 else 0.0
        z_values.append(z)
        print(f"  {name}:")
        print(f"    x = {x},  mean = {mu:.2f},  std = {sigma:.2f}")
        print(f"    z = (x - mean) / std = {z:.4f}")

    # Feature vector in the same order as the model expects
    # (ColumnTransformer: first numeric, then one-hot)
    feature_names = list(preprocessor.get_feature_names_out())
    X_manual = np.array(z_values + [azure_dummy, gcp_dummy], dtype=np.float64)

    print("\n--- 4) FEATURE VECTOR (order the model sees) ---")
    for name, val in zip(feature_names, X_manual):
        print(f"  {name}: {val:.4f}")

    print("\n--- 5) LINEAR REGRESSION EQUATION ---")
    print("  predicted_cost = intercept + (coef_1 * feat_1) + (coef_2 * feat_2) + ...")
    print()
    intercept = float(model.intercept_)
    coefs = model.coef_
    terms = [f"({c:.2f} × {v:.4f})" for c, v in zip(coefs, X_manual)]
    manual_pred = intercept + np.dot(coefs, X_manual)

    print(f"  intercept = {intercept:.2f}")
    for name, c, v in zip(feature_names, coefs, X_manual):
        print(f"  + {c:+.2f} × {v:.4f}  ({name})")
    print(f"  = {intercept:.2f} + sum of terms above")
    print(f"  = {manual_pred:.2f}")

    # Verify with pipeline
    row_df = pd.DataFrame([{k: v for k, v in row.items() if k != TARGET_COLUMN}])
    X_pipeline = preprocessor.transform(row_df)
    pipeline_pred = float(model.predict(X_pipeline)[0])

    print("\n--- 6) VERIFICATION ---")
    print(f"  Prediction (manual):   {manual_pred:.2f}")
    print(f"  Prediction (pipeline): {pipeline_pred:.2f}")
    print(f"  Actual cost (from CSV): {actual_cost:.2f}")
    assert np.isclose(manual_pred, pipeline_pred), "Manual and pipeline should match!"
    print("  [OK] Manual and pipeline match.")

    print("\n" + "=" * 60)
    print("Takeaway: the model is just a weighted sum of standardized usage")
    print("plus provider dummies (vs AWS). No black box.")
    print("=" * 60)


if __name__ == "__main__":
    main()
