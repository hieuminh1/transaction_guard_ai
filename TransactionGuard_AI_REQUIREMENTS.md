# TransactionGuard AI — Detailed Requirements & Junior Developer Implementation Guide

> **Version:** 2.0 | **Language:** Vietnamese (code/API in English) | **Status:** Specification, not implemented | **Target:** Portfolio AI Engineer / financial fraud detection | **Estimated effort:** 14–21 working days for a junior developer, depending on experience.
>
> **How to use:** Read sections 0–5 first, implement tickets in section 17 sequentially, and check each ticket's Definition of Done (DoD). Do not claim any model metric until actually measured.

---

## 0. Quick start: what are we building?

**Business story:** A simulated payment platform receives transaction records. TransactionGuard AI analyzes each record and returns a fraud probability, risk score (0–100), risk level, and flag for manual review. Operators can inspect historical predictions in a basic dashboard.

**Example:** Transaction `txn-1001` of type `TRANSFER`, amount `12000`, sender balance `15000` is sent to `/predict`. API validates data, applies the **same fitted transformations as training**, runs a saved classifier, and returns a structured response. A high risk score means **review**, not automatically blocking a real payment.

**Portfolio evidence:** (1) Data cleaning/EDA; (2) leakage-safe feature engineering; (3) imbalanced classification and model comparison; (4) Precision/Recall/F1/PR-AUC; (5) FastAPI inference; (6) PostgreSQL history; (7) Docker, tests and optional MLflow/dashboard.

### Scope and priorities

| Priority | Included | Not included |
|---|---|---|
| **P0 / MVP** | Dataset download/validation, EDA, time-aware split, Logistic Regression and Random Forest/XGBoost comparison, threshold selection, test report, saved pipeline, `/health`, `/predict`, `/batch-predict`, `/model-info`, unit tests, README | Real bank integration, production transaction blocking |
| **P1 / Portfolio polish** | PostgreSQL prediction logs, `/predictions`, Docker Compose, MLflow, Streamlit dashboard, inference latency metrics | Distributed streaming / Kafka |
| **P2 / Advanced** | Isolation Forest comparison, drift analysis, calibration, GitHub Actions, explanation report | Claims of production-grade fraud performance |

**MVP acceptance:** A new developer can clone, download the dataset, train models, reproduce metrics, launch API and call `/predict` following README commands.

---

## 1. Domain concepts (must understand)

- **Transaction:** one payment/money movement record.
- **Label / target:** `isFraud` (1 = fraud, 0 = legitimate). Only used for supervised training and offline evaluation; **never** accepted as a prediction input.
- **Feature:** an input variable available at prediction time, such as amount, type or sender's *pre-transaction* balance.
- **Inference:** applying a trained model to new transaction data.
- **Probability:** output of `predict_proba` for class 1; it is a model score, not guaranteed calibrated probability.
- **Decision threshold:** if fraud probability >= threshold, mark `is_flagged=true`.
- **Risk score:** `round(probability * 100)` for MVP; explicitly state that this is a scaled model score, **not** a validated business risk rating.
- **False positive (FP):** legitimate transaction flagged as fraud; increases review burden.
- **False negative (FN):** fraudulent transaction not flagged; may cause loss.
- **Data leakage:** model learns from data unavailable when a real prediction is made (e.g., future balances, labels, global aggregates fitted on test data).
- **PR-AUC:** area under precision–recall curve. Use `average_precision_score` and label it **Average Precision (AP)**; do not falsely equate it to trapezoidal PR-AUC.

### Why this project is difficult

Fraud is rare. A classifier predicting every transaction as legitimate can show very high accuracy while catching zero fraud. Therefore, **accuracy is not the selection metric**. The main goal is a transparent precision/recall trade-off evaluated without leakage.

---

## 2. Dataset contract and safe assumptions

### 2.1 Primary dataset: PaySim

Source: <https://www.kaggle.com/datasets/ealaxi/paysim1> (synthetic financial transactions). The user must download it in accordance with the source's terms. Never commit the complete dataset to GitHub.

Expected fields (verify names and types against downloaded CSV):

| Column | Meaning | Training use | API use |
|---|---|---|---|
| `step` | Simulation time step | Yes, with time-split care | Yes |
| `type` | CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER | Yes | Yes |
| `amount` | Transaction amount | Yes | Yes |
| `nameOrig` | Sender identifier | **No in MVP** | No |
| `oldbalanceOrg` | Sender balance before transaction | Yes | Yes |
| `newbalanceOrig` | Sender balance after transaction | **No in MVP**: post-event leakage risk | No |
| `nameDest` | Recipient identifier | **No in MVP** | No |
| `oldbalanceDest` | Recipient balance before transaction | **No in MVP**: simplify availability assumptions | No |
| `newbalanceDest` | Recipient balance after transaction | **No**: leakage risk | No |
| `isFraud` | Target label | Yes, target only | **Never** |
| `isFlaggedFraud` | Simulation/rule-based flag | **No**: target proxy risk | **Never** |

**Important:** The allowed feature set is a **design assumption for this demo**, not proof of production availability. In a real bank, availability must be checked with product/data teams. Document this assumption in README.

### 2.2 Dataset loading and validation

Create `ml/data.py` with `load_dataset(path: Path) -> pd.DataFrame` and `validate_dataset(df) -> None`.

Validation rules:

1. File exists, readable CSV, non-empty.
2. Required columns `step,type,amount,oldbalanceOrg,isFraud` present; optionally inspect all expected source columns.
3. `amount >= 0`, `oldbalanceOrg >= 0`, `step >= 0`, all numeric fields finite.
4. `type` belongs to observed PaySim categories (define a constant list, validate against it).
5. `isFraud` only 0 or 1; both classes exist in training data.
6. Report missing values, duplicate rows, class counts and fraud percentage; don't silently drop rows.
7. Never log raw `nameOrig`/`nameDest` or full raw rows in API logs.

Failure behavior: raise descriptive `ValueError` for missing columns/invalid values, print actionable error and exit nonzero. For huge CSV, read only necessary columns with `usecols` and explicit dtypes if memory is limited.

### 2.3 Train / validation / test split

**Recommended default:** sort by `step` ascending; assign first 70% rows to train, next 15% to validation, final 15% to test. Keep rows with the same `step` together in one split (derive boundary on unique steps or enforce group boundaries). Record step ranges and class counts for each partition.

**Critical edge case:** If validation/test has zero positive labels, STOP and explain why; do not silently randomize. Document an alternative group-aware split only if justified. `test` is locked: never use it for hyperparameters, feature selection or threshold tuning.

Persist split metadata to `reports/split_summary.json` with row counts, positive counts, fraud rates and time ranges. Use the same split for every supervised model.

### 2.4 Alternative dataset

Credit Card Fraud Detection: <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>. It has anonymized V1–V28 features and different API schema. **Do not mix both datasets into the same model/API contract in MVP.** Treat it as an independent benchmark later.

---

## 3. Feature engineering and preprocessing

### 3.1 MVP feature schema

Input raw fields: `step`, `type`, `amount`, `oldbalanceOrg`.

Derived features:

- `amount_to_balance_ratio = amount / max(oldbalanceOrg, 1.0)`; cap extreme ratios at a **documented constant** or use a stable transform (e.g. `log1p`) decided on training data.
- `is_large_amount`: optional flag using a cutoff computed from **training data only** (e.g. train 95th percentile); cutoff must be saved with model.
- `log_amount = log1p(amount)` and `log_oldbalance = log1p(oldbalanceOrg)`.

**Avoid:** features from `newbalanceOrig`, `newbalanceDest`, `isFlaggedFraud`, `isFraud`, full-data fraud statistics, or per-user historical aggregates unless historical information is genuinely available at inference.

### 3.2 Preprocessing pipeline

Implement in `ml/features.py` and `ml/preprocessing.py`:

1. Select approved columns only.
2. Validate numeric ranges and categorical values.
3. Create derived features in a deterministic transformer (`fit`/`transform` compatible).
4. Apply `OneHotEncoder(handle_unknown='ignore')` to `type`.
5. Apply `SimpleImputer(strategy='median')` and `StandardScaler` to numeric columns **if required by the model**; fit on training data only.
6. Use `ColumnTransformer` + `Pipeline` so inference never recreates transformations manually.
7. Serialize the **entire fitted pipeline**, not just classifier, with `joblib`.

**Security:** `joblib`/pickle artifacts must be loaded only from trusted local builds; never deserialize user-uploaded model files.

### 3.3 Consistency tests

- Same input yields same transformed output before/after artifact save-load.
- Column order is fixed and recorded.
- Unknown `type` is rejected by API validation (rather than silently one-hot encoding as all zeros).
- `amount=0`, `oldbalanceOrg=0`, very large finite amount, and missing required field are tested.
- No fitted transformer uses validation/test data.

---

## 4. Model training specification

### 4.1 Algorithms

| Model | Role | Notes |
|---|---|---|
| Logistic Regression | Required baseline | Use scaling; start with `class_weight='balanced'`; fixed seed |
| Random Forest | Required comparison | Try `class_weight='balanced_subsample'`; avoid overly large trees |
| XGBoost | Optional alternative/comparison | Set `scale_pos_weight` from training labels only; record library version |
| Isolation Forest | P2 experiment | Anomaly score is **not** supervised fraud probability |

**Minimum:** 2 supervised models, same train/validation/test partitions and same allowed raw features. Avoid adding SMOTE until baseline is correct. If using SMOTE, place it in `imblearn.pipeline.Pipeline` and resample **train only**, never validation/test.

### 4.2 Training CLI

Implement runnable commands (exact names are a proposed contract):

```bash
python -m ml.train --data data/raw/paysim.csv --output artifacts/
python -m ml.evaluate --artifact artifacts/model.joblib --data data/raw/paysim.csv --split reports/split_summary.json
```

**Implementation detail:** split summary must include reproducible boundary information, not only counts; preferably persist row indices/step cutoffs and an input file checksum. `evaluate` must reconstruct the **exact** held-out test set from saved split metadata. Never rerandomize at evaluation.

Training output:

```text
artifacts/
  model.joblib              # trusted local serialized fitted pipeline
  metadata.json             # model name, version, features, threshold, seed, timestamps
reports/
  eda_summary.json
  split_summary.json
  model_comparison.csv
  validation_metrics.json
  test_metrics.json
  confusion_matrix.png
  precision_recall_curve.png
```

Ignore model binaries and full datasets in `.gitignore` unless artifacts are demonstrably small and license-safe. README should show how to regenerate them.

### 4.3 Selection and threshold policy

1. Train candidate models on train only.
2. Compute **validation Average Precision (AP)** using probability scores.
3. Select best model by highest validation AP; if close, prefer simpler model and report rationale.
4. Choose a decision threshold on validation predictions. MVP policy: among thresholds with `precision >= 0.20` (example **demo-only target**), choose the threshold with highest recall. If no threshold satisfies the constraint, report failure and use a documented fallback (e.g. best validation F1); never pretend target met.
5. Freeze model and threshold.
6. Evaluate once on test; publish actual metrics. Do not retune after seeing test.

Threshold `0.5` is a baseline, **not** automatically correct for imbalanced fraud data. The `0.20` precision target is a configurable exercise setting, not a banking business requirement.

### 4.4 Metrics

Compute with scikit-learn:

```python
precision_score(y_true, y_pred, zero_division=0)
recall_score(y_true, y_pred, zero_division=0)
f1_score(y_true, y_pred, zero_division=0)
average_precision_score(y_true, y_score)
confusion_matrix(y_true, y_pred, labels=[0, 1])
precision_recall_curve(y_true, y_score)
```

Report: model, split, rows, fraud prevalence, threshold, TP/FP/TN/FN, Precision, Recall, F1, AP, and inference latency benchmark. Explain why a model can have high recall but low precision. **Do not invent target accuracy or performance numbers.**

---

## 5. Inference and risk policy

### 5.1 Risk mapping

```text
fraud_probability = float(pipeline.predict_proba(X)[0, 1])
risk_score = round(100 * fraud_probability)
is_flagged = fraud_probability >= metadata.decision_threshold
risk_level = low if risk_score < 30
             medium if 30 <= risk_score < 70
             high if risk_score >= 70
```

Risk bands `30/70` are **display-only demo defaults**, independent of model flag threshold. A record can be `risk_level=medium` but `is_flagged=true`; document this. If calibration has not been performed, call the output a **model-derived score** in UI, not an actuarial probability.

### 5.2 Artifact loading

- On application startup, read artifact path from `MODEL_PATH` env (default `artifacts/model.joblib`).
- Read `metadata.json`; verify required keys and schema version.
- If missing/corrupt artifact, `/health` reports unavailable and `/predict` returns HTTP 503; never produce random fallback predictions.
- Model is loaded once per process and reused.
- Never retrain on API startup.

---

## 6. REST API contract (FastAPI)

**Base URL:** `http://localhost:8000` | **Docs:** `/docs` | **JSON:** UTF-8.

### 6.1 Shared request schema

```json
{
  "transaction_id": "txn-1001",
  "step": 42,
  "type": "TRANSFER",
  "amount": 12000.0,
  "oldbalanceOrg": 15000.0
}
```

| Field | Type | Validation |
|---|---|---|
| `transaction_id` | string | Required, 1–64 characters, non-blank; client-generated ID |
| `step` | integer | Required, >= 0 |
| `type` | enum | `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER` |
| `amount` | finite float | Required, >= 0; reject NaN/Infinity |
| `oldbalanceOrg` | finite float | Required, >= 0; reject NaN/Infinity |

Use Pydantic `extra='forbid'`: reject target labels or accidental unsupported fields. Financial amounts are demo floats because PaySim is numeric; in real money systems prefer integer minor units/Decimal.

### 6.2 `GET /health`

Success 200:

```json
{"status":"ok","model_loaded":true,"model_version":"v1"}
```

Unavailable 503:

```json
{"status":"unavailable","model_loaded":false,"detail":"Model artifact not found"}
```

### 6.3 `POST /predict`

Request: schema above. Success 200 (**illustrative values only**):

```json
{
  "transaction_id": "txn-1001",
  "fraud_probability": 0.82,
  "risk_score": 82,
  "risk_level": "high",
  "is_flagged": true,
  "decision_threshold": 0.61,
  "model_version": "v1"
}
```

Requirements: response probability in `[0,1]`, score integer `[0,100]`, risk enum, deterministic given same artifact/input; no label in response. `decision_threshold` must come from artifact metadata, not hard-coded API constant.

### 6.4 `POST /batch-predict`

Request:

```json
{"transactions":[
  {"transaction_id":"txn-1001","step":42,"type":"TRANSFER","amount":12000,"oldbalanceOrg":15000},
  {"transaction_id":"txn-1002","step":43,"type":"PAYMENT","amount":20,"oldbalanceOrg":200}
]}
```

Response 200:

```json
{"count":2,"predictions":[{"transaction_id":"txn-1001","fraud_probability":0.82,"risk_score":82,"risk_level":"high","is_flagged":true,"decision_threshold":0.61,"model_version":"v1"},{"transaction_id":"txn-1002","fraud_probability":0.03,"risk_score":3,"risk_level":"low","is_flagged":false,"decision_threshold":0.61,"model_version":"v1"}]}
```

Maximum 100 transactions per call (configurable); reject empty list and duplicates in same batch with 422. Predict as a vectorized batch, not a Python loop calling model per row.

### 6.5 `GET /model-info`

```json
{
  "model_version":"v1",
  "algorithm":"RandomForestClassifier",
  "schema_version":"1.0",
  "decision_threshold":0.61,
  "feature_names":["step","type","amount","oldbalanceOrg"],
  "trained_at":"2026-10-24T00:00:00Z",
  "dataset":"PaySim synthetic"
}
```

Metadata example is illustrative; use actual trained model, date and threshold.

### 6.6 Error handling

| Situation | HTTP | Response |
|---|---|---|
| Missing field, negative amount, unsupported type, extra field | 422 | FastAPI/Pydantic structured validation error |
| Missing model | 503 | Stable JSON error detail |
| Unexpected internal exception | 500 | Generic safe message; full stack trace server-side only |
| Oversized batch | 422 | Documented max size |
| Unknown endpoint | 404 | Standard FastAPI response |

Never expose secrets, raw traceback or internal filesystem paths in API error responses.

### 6.7 Curl examples

```bash
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"transaction_id":"txn-1001","step":42,"type":"TRANSFER","amount":12000,"oldbalanceOrg":15000}'
```

---

## 7. PostgreSQL history (P1)

**Purpose:** Keep metadata of model predictions for demo monitoring. Not required for model training.

Suggested table:

```sql
CREATE TABLE prediction_logs (
    id BIGSERIAL PRIMARY KEY,
    transaction_id VARCHAR(64) NOT NULL,
    model_version VARCHAR(40) NOT NULL,
    transaction_type VARCHAR(16) NOT NULL,
    amount DOUBLE PRECISION NOT NULL CHECK (amount >= 0),
    fraud_probability DOUBLE PRECISION NOT NULL CHECK (fraud_probability BETWEEN 0 AND 1),
    risk_score SMALLINT NOT NULL CHECK (risk_score BETWEEN 0 AND 100),
    risk_level VARCHAR(10) NOT NULL,
    is_flagged BOOLEAN NOT NULL,
    predicted_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_prediction_logs_predicted_at ON prediction_logs (predicted_at DESC);
CREATE INDEX idx_prediction_logs_flagged ON prediction_logs (is_flagged, predicted_at DESC);
```

Do **not** add unique constraint on `transaction_id` alone: repeat scoring with a different model version may be useful. Add `GET /predictions?limit=20&offset=0&flagged=true` (limit 1–100) returning `items`, `total`, `limit`, `offset`. Use SQLAlchemy + Alembic; all DB credentials via environment variables. If DB unavailable, decide and document fail-open (predict succeeds but log warning) vs fail-closed; **recommended for demo:** prediction succeeds, logging failure is observable but does not change model output.

---

## 8. Dashboard (P1)

Choose **Streamlit** for junior-friendly scope; React is optional if already familiar. Dashboard is read-only, no real payment actions.

Screens:

1. **Overview:** total scored, flagged count, flagged rate, average model score, latest model version.
2. **Predictions:** sortable/paginated table with transaction ID, type, amount, score, level, flag, timestamp.
3. **Model Evaluation:** static test-set AP, Precision, Recall, F1, confusion matrix and PR curve; clearly label as **offline held-out test metrics**, not live monitoring.
4. **Try a transaction:** form calls `/predict` and displays response; example values explicitly synthetic.

Acceptance: dashboard runs locally, never exposes passwords, handles API unavailable state gracefully, and clearly distinguishes historical labels (if available) from model predictions.

---

## 9. Monitoring (P1/P2)

**P1 operational monitoring:** request count, request latency p50/p95 (from observed requests), errors, flag rate, risk-score distribution, model version. Logs must not include sender identifiers.

**P2 drift:** compare current incoming feature distributions against training baseline using PSI or another documented statistic; set demo thresholds explicitly as **heuristics**, not universal guarantees. Store baseline histogram/bin edges from train only. Avoid false claims that drift proves model degradation.

**Ground truth caveat:** Real Precision, Recall, FP and FN in production require **later verified fraud labels**. They cannot be computed from prediction logs alone. A change in flagged rate is not automatically model drift.

---

## 10. MLflow and model registry (P1)

Log run name, algorithm, seed, training dataset checksum, allowed feature schema, preprocessing version, hyperparameters, train/validation split ranges, validation AP, validation Precision/Recall/F1 at selected threshold, and model artifact path. If logging test metrics, mark them as **final test** and do not use for subsequent tuning.

Use local file-backed MLflow for simplicity; never require a paid cloud service. A model version can be a timestamp or git short SHA. Ensure `metadata.json` and artifact version match.

---

## 11. Repository layout (files to implement)

```text
transactionguard-ai/
├── README.md                         # project overview + exact run commands
├── REQUIREMENTS.md                   # this specification
├── .gitignore                        # raw data, .env, artifacts, caches
├── .env.example                      # safe variable names/defaults
├── pyproject.toml                    # pinned or constrained dependencies + lint config
├── Dockerfile
├── docker-compose.yml                # P1: api + postgres + optional dashboard
├── data/
│   ├── raw/.gitkeep                  # dataset downloaded locally, ignored by git
│   └── sample/sample_transactions.json # handcrafted synthetic API fixtures
├── app/
│   ├── __init__.py
│   ├── main.py                        # app factory/startup, routes
│   ├── core/config.py                 # env settings
│   ├── api/routes.py                  # HTTP handlers
│   ├── schemas/transaction.py         # Pydantic input/output models
│   ├── services/predictor.py          # model loading and batch scoring
│   ├── db/models.py                   # P1 SQLAlchemy
│   └── db/session.py                  # P1 DB connection
├── ml/
│   ├── __init__.py
│   ├── data.py                        # dataset validation, splits
│   ├── features.py                    # feature transformer
│   ├── preprocessing.py               # sklearn ColumnTransformer
│   ├── train.py                       # train CLI
│   ├── evaluate.py                    # report CLI
│   ├── metrics.py                     # scoring and threshold selection
│   └── artifact.py                    # serialize model + metadata
├── configs/
│   └── training.yaml                  # seed, features, model params
├── notebooks/
│   └── 01_eda.ipynb
├── reports/                           # generated summaries/figures
├── artifacts/                         # generated trained models; gitignored
├── dashboard/                          # P1 Streamlit
├── migrations/                         # P1 Alembic
├── tests/
│   ├── conftest.py
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_metrics.py
│   ├── test_predictor.py
│   └── test_api.py
└── .github/workflows/ci.yml            # P2
```

**Dependency groups:** core (`pandas`, `numpy`, `scikit-learn`, `fastapi`, `uvicorn`, `pydantic`, `joblib`, `matplotlib`, `pytest`, `httpx`); optional (`xgboost`, `imbalanced-learn`, `mlflow`, `sqlalchemy`, `psycopg`, `alembic`, `streamlit`). Lock or constrain versions and record Python version (recommend Python 3.11 or 3.12). Avoid importing P1-only libraries into P0 runtime.

---

## 12. Environment setup (junior checklist)

Prerequisites: Python 3.11+, Git, 4–8 GB available RAM (dataset/model size dependent), Docker optional for MVP.

```bash
git clone <YOUR_REPOSITORY_URL>
cd transactionguard-ai
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e '.[dev]'
cp .env.example .env                 # Windows: copy .env.example .env
# Download PaySim from Kaggle manually to data/raw/paysim.csv
python -m ml.train --data data/raw/paysim.csv --output artifacts/
pytest -q
uvicorn app.main:app --reload
# Visit http://localhost:8000/docs
```

**Note:** These are **target commands** the junior developer must make work; the repo is not yet implemented. Use actual downloaded filename or document a rename step.

Suggested `.env.example`:

```dotenv
APP_ENV=development
MODEL_PATH=artifacts/model.joblib
MODEL_METADATA_PATH=artifacts/metadata.json
DATABASE_URL=postgresql+psycopg://transactionguard:transactionguard@localhost:5432/transactionguard
LOG_LEVEL=INFO
MAX_BATCH_SIZE=100
```

Never commit `.env` or real passwords. Example DB password is **local demo only**, not for deployment.

---

## 13. Testing requirements (mandatory)

### 13.1 Unit test matrix

| ID | Scenario | Expected |
|---|---|---|
| T-DATA-01 | CSV missing `isFraud` | Descriptive validation error |
| T-DATA-02 | Invalid negative amount | Reject |
| T-DATA-03 | Training/validation/test overlap | Zero overlapping rows/time steps |
| T-DATA-04 | Test has no fraud labels | Training/evaluation stops with explanation |
| T-FEAT-01 | `oldbalanceOrg=0` | Finite ratio, no division by zero |
| T-FEAT-02 | Save/load pipeline | Same prediction for identical input |
| T-FEAT-03 | Target column in input | Never included in model features |
| T-MET-01 | Known labels/predictions | Expected confusion matrix counts |
| T-MET-02 | AP calculation | Matches scikit-learn reference |
| T-MET-03 | Threshold tuning | Uses validation only |
| T-API-01 | Valid POST `/predict` | 200 + correct response schema |
| T-API-02 | Negative amount | 422 |
| T-API-03 | Extra `isFraud` field | 422 |
| T-API-04 | Unsupported transaction type | 422 |
| T-API-05 | Empty batch / >100 rows | 422 |
| T-API-06 | Missing model artifact | 503 |
| T-API-07 | Repeated identical request | Same score with same model |
| T-API-08 | `/model-info` | Correct artifact version and threshold |

### 13.2 Testing strategy

- Unit tests use **tiny synthetic in-memory data**, not the entire Kaggle dataset.
- `TestClient` with dependency injection/mock predictor for API tests; tests must run without a trained artifact or DB.
- Separate integration test runs full train → save → load → predict on a small fixture.
- No network access required for unit tests.
- Keep an explicit `pytest -q` command and fix all failures before merge.

### 13.3 Quality and reproducibility

Use `ruff check .`, `ruff format --check .` (or equivalent), typed function signatures for public modules, docstrings for non-obvious transforms, stable seeds, and a dataset checksum. Set expectations honestly: stochastic algorithms may differ across library versions; document environment.

---

## 14. Security, privacy and non-functional requirements

| ID | Requirement | Acceptance |
|---|---|---|
| NFR-01 | No real financial credentials or private user data | No secrets in repo or logs |
| NFR-02 | Safe deserialization | Only load trusted local joblib artifacts |
| NFR-03 | Input validation | Reject malformed values and unknown fields |
| NFR-04 | Reproducibility | README commands generate reports and artifact |
| NFR-05 | API responsiveness | Record measured local latency; no invented SLA |
| NFR-06 | Error handling | Missing model gives 503; no random fallback |
| NFR-07 | Documentation | Each CLI has `--help` and examples |
| NFR-08 | Dataset license compliance | Source linked; raw dataset excluded from repo |
| NFR-09 | Honest scope | README states synthetic data and no real bank integration |
| NFR-10 | Review safety | Flag means manual review, not automated payment rejection |

**Out of scope for MVP:** authentication, rate limiting, audit compliance, encryption-at-rest and real production access control. If publicly deployed, these become required before exposing any sensitive data.

---

## 15. Architecture and data flow

```mermaid
flowchart TD
  A[PaySim CSV] --> B[Validate + EDA]
  B --> C[Chronological split]
  C --> D[Fit features on train]
  D --> E[Train LR + RF]
  E --> F[Validate + select threshold]
  F --> G[Final test evaluation]
  G --> H[Save pipeline + metadata]
  H --> I[FastAPI predictor]
  J[Transaction JSON] --> I
  I --> K[Risk score + flag]
  K --> L[(PostgreSQL - P1)]
  L --> M[Dashboard - P1]
```

**Offline:** data → split → fit preprocessing on train → train → validate → select threshold → final test → save artifact. **Online:** validate JSON → transform using saved pipeline → score → map risk → respond → optionally log. Never put training in API request path.

---

## 16. API and model metadata examples

Suggested `artifacts/metadata.json` shape:

```json
{
  "schema_version": "1.0",
  "model_version": "v1",
  "algorithm": "RandomForestClassifier",
  "dataset_name": "PaySim synthetic",
  "dataset_sha256": "<actual SHA-256 after download>",
  "trained_at": "<actual ISO-8601 UTC timestamp>",
  "random_seed": 42,
  "raw_feature_names": ["step", "type", "amount", "oldbalanceOrg"],
  "decision_threshold": 0.61,
  "risk_bands": {"low_max_exclusive": 30, "medium_max_exclusive": 70},
  "validation_average_precision": null,
  "test_average_precision": null
}
```

The threshold `0.61` is an **example only**. Code must replace it with the selected validation threshold; replace `null` metrics with measured values. Validate metadata at startup; fail fast if schema incompatible.

---

## 17. Step-by-step development tickets (assign to a junior)

Each ticket must be committed separately, have tests and a short PR description. Suggested sequence is dependency-aware.

### Sprint 1 — Data and baseline (days 1–4)

#### TG-01: Initialize repository (0.5 day)

**Tasks:** Create layout from section 11, `pyproject.toml`, `.gitignore`, `.env.example`, basic README, pre-commit/ruff config. Add `python -m ml.train --help` placeholder or real CLI parser.

**DoD:** Fresh venv installs; `pytest -q` runs; `.env` and `data/raw/*.csv` ignored; README states synthetic-only.

#### TG-02: Download and validate dataset (0.5–1 day)

**Tasks:** Document Kaggle URL/download/rename, implement `load_dataset`, schema validator, create `reports/eda_summary.json` with class counts/missing/duplicates.

**DoD:** Works on downloaded CSV; invalid fixture fails clearly; tests T-DATA-01/02 pass.

#### TG-03: EDA notebook (0.5 day)

**Tasks:** Fraud rate, transaction-type distribution, amount distribution (use log axis if helpful), missing values, and leakage notes. Export figures to `reports/`.

**DoD:** Notebook runs top-to-bottom on local dataset; all plots have labels; summary describes class imbalance and synthetic limitations.

#### TG-04: Chronological split (0.5–1 day)

**Tasks:** Implement split by `step` groups, preserve test isolation, persist boundaries/checksum and class counts.

**DoD:** No overlapping step values; all partitions contain positives or fail clearly; reproducible on same CSV.

#### TG-05: Feature pipeline (1 day)

**Tasks:** Implement deterministic derived features, OneHotEncoder and numeric preprocessing in a scikit-learn Pipeline. Never use forbidden columns.

**DoD:** Pipeline fit on train only; save/load produces same output; zero-balance and unknown category cases tested.

### Sprint 2 — ML evaluation and artifact (days 5–8)

#### TG-06: Logistic Regression baseline (0.5–1 day)

**Tasks:** Train baseline; log params and validation scores; use class weights; create initial comparison CSV.

**DoD:** Train CLI works; results reproducible; no test data used for selection.

#### TG-07: Random Forest or XGBoost (1 day)

**Tasks:** Train second model on identical split, compare validation AP and compute metrics at threshold 0.5.

**DoD:** Comparison table with actual values; record runtime and model parameters; explain selected candidate.

#### TG-08: Threshold tuning and held-out test (1 day)

**Tasks:** Implement validation-only threshold search, record rule/fallback, freeze selected model/threshold, evaluate test once. Generate confusion matrix and PR curve.

**DoD:** All required metrics published; test is not used for tuning; test fixture verifies confusion matrix.

#### TG-09: Save/reload artifact (0.5 day)

**Tasks:** Serialize entire fitted pipeline and metadata. Add checksum/version; write `ml.predict` function.

**DoD:** Same input produces same probability before/after load; corrupted/missing artifact fails cleanly.

### Sprint 3 — API (days 9–10)

#### TG-10: FastAPI app and schemas (1 day)

**Tasks:** Implement startup model loading, `/health`, `/model-info`, `/predict`, Pydantic schema, risk mapping.

**DoD:** Valid curl succeeds; malformed fields return 422; model missing returns 503; Swagger shows examples.

#### TG-11: Batch inference and API tests (1 day)

**Tasks:** `/batch-predict`, 1–100 records, vectorized scoring, consistent order, duplicate ID rejection; tests.

**DoD:** 100 inputs return 100 ordered outputs; empty/oversized/duplicate rejected; unit/API tests pass.

### Sprint 4 — Portfolio improvements (days 11–14+)

#### TG-12: PostgreSQL logs (1 day, P1)

**Tasks:** SQLAlchemy table, Alembic migration, logging after successful inference, `/predictions` pagination.

**DoD:** Predictions persisted and queried; database outage behavior documented; no private sender identifiers stored.

#### TG-13: Docker Compose (0.5–1 day, P1)

**Tasks:** Multi-stage or lean Dockerfile, Compose API + PostgreSQL, health checks, volumes/config.

**DoD:** `docker compose up --build` starts services after model artifact is created; README shows clean setup.

#### TG-14: MLflow (0.5–1 day, P1)

**Tasks:** Log algorithm, parameters, validation metrics and artifact metadata. Document local UI.

**DoD:** At least two comparable runs visible; no sensitive data logged.

#### TG-15: Streamlit dashboard (1–2 days, P1)

**Tasks:** Implement four screens in section 8 with API calls and demo labels.

**DoD:** Screenshot-ready dashboard; empty data and API failure handled; no claim of real-time ground truth.

#### TG-16: README and demo polish (0.5–1 day)

**Tasks:** Architecture diagram, setup commands, sample curl, screenshots, real comparison table, leakage limitations, engineering decisions and future work.

**DoD:** Another developer can follow README without verbal help; repo has no secrets/raw PaySim dataset.

#### TG-17: Optional advanced ML (P2)

**Tasks:** Isolation Forest experiment; compare anomaly scores separately, optional calibration/drift, GitHub Actions CI.

**DoD:** Honest experiment report including negative results; no unsupported fraud probabilities.

---

## 18. Suggested PR review checklist

- [ ] Ticket scope only; no unrelated refactoring.
- [ ] All inputs validated and errors actionable.
- [ ] No target/post-transaction leakage.
- [ ] Train/validation/test separation respected.
- [ ] New logic has unit tests; `pytest -q` passes.
- [ ] `ruff check .` passes.
- [ ] New env vars added to `.env.example`.
- [ ] README updated for any new CLI/API endpoint.
- [ ] No `.env`, secrets, raw dataset or oversized binary committed.
- [ ] Claims/metrics correspond to real measured outputs.

**PR description template:** `Goal`, `Files changed`, `How to run`, `How tested`, `Screenshots/metrics`, `Known limitations`.

---

## 19. Definition of Done — overall

### P0 / required before public GitHub showcase

- [ ] `python -m ml.train ...` works from fresh clone with downloaded dataset.
- [ ] EDA shows fraud prevalence, missing data and dataset limitations.
- [ ] Split boundaries are reproducible and leakage precautions documented.
- [ ] Two supervised models evaluated on same protocol.
- [ ] Validation-based threshold selected; test evaluated only after freeze.
- [ ] Test Precision, Recall, F1, AP, confusion matrix and PR curve generated.
- [ ] Full preprocessing + classifier saved in trusted artifact.
- [ ] `/health`, `/predict`, `/batch-predict`, `/model-info` match documented schema.
- [ ] API rejects bad inputs and fails clearly without artifact.
- [ ] Unit and API tests pass without downloading dataset.
- [ ] README includes commands, architecture, limitations and example output.

### P1 / preferred portfolio finish

- [ ] Prediction history stored in PostgreSQL with migrations.
- [ ] Docker Compose runs API and DB locally.
- [ ] MLflow tracks at least two experiments.
- [ ] Dashboard shows history, flagged transactions and offline evaluation.
- [ ] Basic latency/error/flag-rate monitoring included.

---

## 20. Common junior mistakes and how to avoid them

1. **Using `isFraud` in feature matrix:** model cheats. Always define an explicit allowed-feature list.
2. **Using `newbalanceOrig` or `isFlaggedFraud`:** potentially post-event information/label proxy. Exclude in MVP.
3. **Randomly splitting time-series-like records:** can make evaluation optimistic. Use `step` groups and document limitations.
4. **Running SMOTE before split:** test contamination. Resample training only.
5. **Reporting accuracy only:** misleading with rare fraud. Publish AP, Precision, Recall, F1 and confusion matrix.
6. **Tuning on test set:** invalid final performance. Freeze threshold on validation.
7. **Different preprocessing in API:** scores won't match offline. Save full fitted pipeline.
8. **Hard-coding model threshold in API:** mismatch with selected model. Read from metadata.
9. **Calling score calibrated probability without calibration:** misleading. State model-derived score.
10. **Using live FP rate without labels:** impossible to know actual FP/FN from predictions alone.
11. **Publishing Kaggle CSV or credentials:** privacy/license/security problem. Ignore them in Git.
12. **Adding too many services before baseline works:** slows learning. Complete P0 before P1/P2.

---

## 21. Interview / CV evidence after implementation

When completed, this repo should let you truthfully explain: how you chose safe features; why imbalanced data changes metric selection; how you compared models; how threshold impacts manual review volume; how you prevented train/test leakage; how you packaged preprocessing with inference; how your API validates inputs; and what would need to change for a real financial system.

**Suggested CV wording only after actual completion:**

> Developed an end-to-end fraud-detection ML pipeline on synthetic financial transactions, comparing supervised classifiers, evaluating imbalanced performance with Precision/Recall/F1/Average Precision, and deploying reproducible risk-scoring inference through FastAPI.

**Do not claim:** real banking production usage, specific model metrics you haven't measured, fraud losses prevented, calibrated risk probability, or model drift/false-positive monitoring without verified ground-truth labels.

---

## 22. References and learning links

- PaySim dataset: <https://www.kaggle.com/datasets/ealaxi/paysim1>
- Alternative credit card dataset: <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>
- Scikit-learn pipelines: <https://scikit-learn.org/stable/modules/compose.html>
- Precision/recall metrics: <https://scikit-learn.org/stable/modules/model_evaluation.html>
- Imbalanced-learn: <https://imbalanced-learn.org/stable/>
- FastAPI tutorial: <https://fastapi.tiangolo.com/tutorial/>
- MLflow tracking: <https://mlflow.org/docs/latest/ml/tracking/>
- Pydantic: <https://docs.pydantic.dev/latest/>

**Final reminder:** This document is an implementation plan. Commands, files, example scores and example timestamps describe what to build; they do **not** indicate the project is already implemented or that any metric has been achieved.
