from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.config import CLOUD_PROVIDER_CATEGORIES


NUMERIC_FEATURES = [
    "cpu_hours",
    "storage_gb",
    "bandwidth_gb",
    "active_users",
    "uptime_hours",
]

CATEGORICAL_FEATURES = ["cloud_provider"]


def build_preprocessor() -> ColumnTransformer:
    """
    WHAT:
      Returns a fitted-on-data later "preprocessor" that converts a raw DataFrame
      into a numeric matrix ready for Linear Regression.

    WHY:
      - Linear Regression expects numeric inputs.
      - We want AWS as the baseline provider (reference category) so the provider
        coefficients become "difference vs AWS".
      - We standardize numeric features so coefficients are comparable and
        optimization is stable.

    HOW:
      - OneHotEncoder turns cloud_provider into dummy variables.
        With categories fixed to [AWS, Azure, GCP] and drop='first', we drop AWS,
        creating columns:
          cloud_provider_Azure, cloud_provider_GCP
        meaning:
          +1 when provider is that cloud, 0 otherwise.
      - StandardScaler applies z-score scaling: (x - mean) / std.
    """
    provider_encoder = OneHotEncoder(
        categories=[CLOUD_PROVIDER_CATEGORIES],
        drop="first",  # drop AWS -> AWS becomes the baseline
        handle_unknown="ignore",
        sparse_output=False,
    )

    numeric_scaler = StandardScaler()

    return ColumnTransformer(
        transformers=[
            ("num", numeric_scaler, NUMERIC_FEATURES),
            ("provider", provider_encoder, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def to_dataframe_row(payload: dict) -> pd.DataFrame:
    """
    Utility for API/UI: turn a single JSON-like dict into a 1-row DataFrame
    with expected columns.
    """
    return pd.DataFrame([payload])

