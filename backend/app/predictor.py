from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from ml.preprocess import to_dataframe_row


@dataclass(frozen=True)
class Predictor:
    model: object
    preprocessor: object

    def predict_one(self, payload: dict) -> float:
        """
        WHAT:
          Predict monthly cloud cost for one request.

        WHY:
          Keep model logic out of FastAPI route handlers (clean architecture).

        HOW:
          - Convert JSON -> 1-row DataFrame
          - Apply the SAME transformations used in training
          - Run linear regression prediction
        """
        X = to_dataframe_row(payload)
        X_t = self.preprocessor.transform(X)
        pred = self.model.predict(X_t)
        return float(np.asarray(pred).ravel()[0])


def load_predictor(artifacts_dir: Path) -> Predictor:
    model_path = artifacts_dir / "model.pkl"
    preprocessor_path = artifacts_dir / "scaler.pkl"

    if not model_path.exists() or not preprocessor_path.exists():
        raise FileNotFoundError(
            "Missing artifacts. Train the model first.\n"
            f"Expected: {model_path} and {preprocessor_path}"
        )

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return Predictor(model=model, preprocessor=preprocessor)

