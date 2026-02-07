## Multi-Cloud Cost Prediction System

Predict monthly cloud cost from resource usage + cloud provider (AWS/Azure/GCP) using **Linear Regression**, served via **FastAPI**, and consumed by **Streamlit**.

---

## Folder structure (clean + interview-ready)

- `data/`
  - Put your CSV here (example: `data/cloud_costs.csv`)
- `ml/`
  - `train.py`: trains model, evaluates, saves artifacts
  - `preprocess.py`: one-hot + scaling (same logic used by API)
  - `config.py`: feature list and fixed provider category order
- `artifacts/`
  - `model.pkl`: trained `LinearRegression`
  - `scaler.pkl`: fitted preprocessor (**scaler + one-hot encoder**, kept name per requirement)
  - `training_meta.pkl`: metrics + coefficients for inspection
- `backend/`
  - `app/main.py`: FastAPI app + routes
  - `app/schemas.py`: Pydantic request/response models
  - `app/predictor.py`: loads artifacts and predicts
- `frontend/`
  - `app.py`: Streamlit UI calling the backend

---

## Step-by-step (what/why/how)

### 1) One-Hot Encoding (AWS as reference)

- **WHAT**: Convert `cloud_provider` (AWS/Azure/GCP) into numeric columns.
- **WHY**: Regression needs numbers; categories must become indicators.
- **HOW**:
  - We fix the category order to `["AWS", "Azure", "GCP"]`.
  - We set `drop="first"` so **AWS is dropped** and becomes the **baseline**.
  - The model gets two columns:
    - `cloud_provider_Azure` (1 if Azure else 0)
    - `cloud_provider_GCP` (1 if GCP else 0)

**Interpretation**: The Azure/GCP coefficients represent *how much more (or less) cost you expect compared to AWS*, holding usage constant.

### 2) Feature scaling (Standardization / Z-score)

- **WHAT**: Convert each numeric feature into \(z = (x - \mu) / \sigma\).
- **WHY scaling is needed (industry-correct intuition)**:
  - Your features have different units and ranges (GB vs hours vs users).
  - Scaling makes coefficients more comparable: “1 unit” becomes “1 standard deviation”.
  - It improves numerical stability (especially important as projects grow to regularized models).
- **HOW**:
  - During training we compute mean/std on **training data only**
  - Then we apply the same mean/std to every future request (API/Streamlit)

### 3) Linear Regression (what the model is doing)

The model predicts:

\[
\hat{y} = b_0 + \sum_i b_i x_i
\]

After preprocessing, \(x_i\) are:

- standardized usage features (z-scores)
- provider dummy variables vs AWS baseline

**What a coefficient means here**

- For standardized numeric features: coefficient ≈ change in cost for a **+1 standard deviation** increase in that usage, holding others constant.
- For provider dummies: coefficient ≈ difference in cost between that provider and **AWS**, holding usage constant.

---

## Run the project (end-to-end)

### 0) Setup environment

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 1) Put your dataset in place

Save your CSV as:

- `data/cloud_costs.csv`

### 2) Train + save artifacts

```powershell
.\run_train.ps1
```
Or: `python -m ml.train --csv data/multicloud_cost_prediction_dataset.csv`

This creates `artifacts/model.pkl` and `artifacts/scaler.pkl`.

### 3) Start FastAPI

```powershell
.\run_backend.ps1
```
Or: `python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000`

Check:

- `GET http://localhost:8000/health`
- `GET http://localhost:8000/evaluation` (returns MAE, RMSE, R²)
- Open http://localhost:8000/docs to see all routes (including `/evaluation`).

**If the UI shows "Evaluation: API returned 404"**: the backend running is an old process. Stop it (Ctrl+C in the backend terminal), then run `.\run_backend.ps1` again so the server loads the current code with the `/evaluation` route.

### 4) Start Streamlit

```powershell
.\run_frontend.ps1
```
Or: `streamlit run frontend/app.py`

---

## Common mistakes to avoid (real-world)

- **Data leakage**: fitting scaler/encoder on the full dataset (train+test) instead of train only.
- **Mismatched preprocessing**: training uses one encoding/scaling, API uses another (guaranteed wrong predictions).
- **Wrong baseline category**: if AWS is not the dropped category, “provider coefficients” won’t mean “vs AWS”.
- **Unseen category handling**: production traffic might send unknown providers; we use `handle_unknown="ignore"` to avoid crashes.
- **Assuming causality**: linear regression learns correlation, not “CPU causes cost”.

---

## Interview-ready talking points

- **Reproducibility**: preprocessing is saved and reused at inference time.
- **Separation of concerns**:
  - training code in `ml/`
  - inference logic in `backend/app/predictor.py`
  - UI in `frontend/`
- **Interpretability**:
  - coefficients map to standardized usage impact
  - provider coefficients are differences vs AWS baseline

