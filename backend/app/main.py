from __future__ import annotations

from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.app.predictor import Predictor, load_predictor
from backend.app.schemas import CostPredictionRequest, CostPredictionResponse


def create_app() -> FastAPI:
    app = FastAPI(title="Multi-Cloud Cost Prediction API", version="1.0.0")

    # Allow Streamlit (and other origins) to call the API from the browser
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    predictor: Predictor | None = None
    project_root: Path = Path(__file__).resolve().parents[2]
    artifacts_dir: Path = project_root / "artifacts"

    @app.on_event("startup")
    def _startup() -> None:
        nonlocal predictor
        predictor = load_predictor(artifacts_dir)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/routes")
    def list_routes() -> dict:
        """List registered routes (useful to confirm /evaluation is loaded)."""
        routes = [r.path for r in app.routes if hasattr(r, "path") and not r.path.startswith("/openapi")]
        return {"routes": sorted(routes)}

    @app.get("/evaluation")
    def evaluation() -> dict:
        """Return model evaluation metrics (MAE, RMSE, R²) from training."""
        meta_path = artifacts_dir / "training_meta.pkl"
        if not meta_path.exists():
            raise HTTPException(status_code=404, detail="Evaluation not available. Train the model first.")
        try:
            meta = joblib.load(meta_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not load evaluation: {e!s}")
        metrics = meta.get("metrics")
        if not metrics:
            raise HTTPException(status_code=404, detail="No metrics in training meta.")
        return metrics

    @app.post("/predict", response_model=CostPredictionResponse)
    def predict(req: CostPredictionRequest) -> CostPredictionResponse:
        assert predictor is not None, "Predictor not loaded"
        pred = predictor.predict_one(req.model_dump())
        return CostPredictionResponse(predicted_monthly_cloud_cost=pred)

    return app


app = create_app()

