from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def artifacts_dir(self) -> Path:
        return self.project_root / "artifacts"

    @property
    def default_csv_path(self) -> Path:
        # You can override this from the CLI: --csv path/to/file.csv
        return self.data_dir / "cloud_costs.csv"


FEATURE_COLUMNS = [
    "cloud_provider",
    "cpu_hours",
    "storage_gb",
    "bandwidth_gb",
    "active_users",
    "uptime_hours",
]

TARGET_COLUMN = "monthly_cloud_cost"

# IMPORTANT: Fix the category order so that One-Hot Encoding with drop='first'
# uses AWS as the reference category (baseline).
CLOUD_PROVIDER_CATEGORIES = ["AWS", "Azure", "GCP"]

