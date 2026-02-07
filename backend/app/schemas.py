from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


CloudProvider = Literal["AWS", "Azure", "GCP"]


class CostPredictionRequest(BaseModel):
    """
    WHAT:
      Defines the JSON shape your API accepts.

    WHY:
      - Input validation prevents nonsense inputs from reaching your model.
      - It's a contract between frontend and backend.
    """

    cloud_provider: CloudProvider

    cpu_hours: float = Field(ge=0)
    storage_gb: float = Field(ge=0)
    bandwidth_gb: float = Field(ge=0)
    active_users: float = Field(ge=0)
    uptime_hours: float = Field(ge=0)


class CostPredictionResponse(BaseModel):
    predicted_monthly_cloud_cost: float

