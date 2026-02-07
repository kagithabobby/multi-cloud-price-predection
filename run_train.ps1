# Train model and save model.pkl + scaler.pkl (from project root).
# Usage: .\run_train.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python -m ml.train --csv data/multicloud_cost_prediction_dataset.csv
