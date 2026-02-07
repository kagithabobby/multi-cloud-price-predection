## FastAPI backend

### Run

From the project root:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Train once first (needs your CSV in data/)
python -m ml.train --csv data/cloud_costs.csv

# Start API
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### Endpoints

- `GET /health`
- `POST /predict`

Example request body:

```json
{
  "cloud_provider": "AWS",
  "cpu_hours": 120,
  "storage_gb": 500,
  "bandwidth_gb": 2000,
  "active_users": 50,
  "uptime_hours": 720
}
```
