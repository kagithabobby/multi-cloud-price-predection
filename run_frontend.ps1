# Run Streamlit frontend (from project root).
# Start the backend first: .\run_backend.ps1
# Usage: .\run_frontend.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python -m streamlit run frontend/app.py
