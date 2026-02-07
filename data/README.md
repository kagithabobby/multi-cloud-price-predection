## Put your dataset here

Place your CSV file in this folder (for example: `data/cloud_costs.csv`).

The CSV must contain these columns:

- `cloud_provider` (AWS / Azure / GCP)
- `cpu_hours`
- `storage_gb`
- `bandwidth_gb`
- `active_users`
- `uptime_hours`
- `monthly_cloud_cost` (target)

This project is structured so training reads the CSV from `data/`, while the API + Streamlit use saved artifacts from `artifacts/`.
