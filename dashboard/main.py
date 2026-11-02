"""Read-only prediction inspection and synthetic transaction scoring."""

import json
import os
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
REPORTS = Path(os.getenv("REPORTS_DIR", "reports"))
st.set_page_config(page_title="TransactionGuard AI", page_icon="🛡️", layout="wide")
st.title("TransactionGuard AI")
st.caption("Synthetic demo · Model-derived scores · Flags request manual review")


def api(path: str, payload: dict | None = None):
    try:
        with httpx.Client(timeout=8) as client:
            response = (
                client.get(API_URL + path)
                if payload is None
                else client.post(API_URL + path, json=payload)
            )
            response.raise_for_status()
            return response.json()
    except (httpx.HTTPError, ValueError):
        st.warning(
            "API or prediction history is unavailable. Check API, model and database status."
        )
        return None


page = st.sidebar.radio(
    "View", ["Overview", "Predictions", "Model Evaluation", "Try a transaction"]
)
if page == "Overview":
    summary = api("/predictions/summary")
    if summary:
        columns = st.columns(4)
        columns[0].metric("Scored", summary["total_scored"])
        columns[1].metric("Flagged", summary["flagged_count"])
        columns[2].metric("Flagged rate", f"{summary['flagged_rate']:.1%}")
        average = summary["average_model_score"]
        columns[3].metric("Average score", f"{average:.1f}" if average is not None else "—")
        st.write("Latest model:", summary["latest_model_version"] or "No predictions yet")
    metrics = api("/metrics")
    if metrics:
        st.subheader("Operational monitoring")
        st.json(metrics)
    st.info(
        "Prediction logs have no verified labels. "
        "Live precision, recall and false positives are unknown."
    )
elif page == "Predictions":
    flagged = st.selectbox("Filter", ["All", "Flagged", "Unflagged"])
    size = st.selectbox("Rows per page", [20, 50, 100])
    number = st.number_input("Page", min_value=1, value=1, step=1)
    query = f"/predictions?limit={size}&offset={(number - 1) * size}"
    if flagged != "All":
        query += f"&flagged={'true' if flagged == 'Flagged' else 'false'}"
    result = api(query)
    if result:
        st.write(f"{result['total']} matching predictions")
        if result["items"]:
            st.dataframe(pd.DataFrame(result["items"]), use_container_width=True, hide_index=True)
        else:
            st.info("No predictions on this page.")
elif page == "Model Evaluation":
    st.subheader("Offline held-out test evaluation")
    try:
        report = json.loads((REPORTS / "test_metrics.json").read_text())
        st.caption(f"Dataset: {report['dataset_name']} · Model: {report['model_version']}")
        columns = st.columns(4)
        for column, key in zip(
            columns, ["average_precision", "precision", "recall", "f1"], strict=True
        ):
            column.metric(key.replace("_", " ").title(), f"{report[key]:.4f}")
        st.json(report)
        for filename in ("confusion_matrix.png", "precision_recall_curve.png"):
            if (REPORTS / filename).is_file():
                st.image(str(REPORTS / filename))
    except (OSError, ValueError, KeyError):
        st.info("No readable test report. Train a model to generate offline evaluation reports.")
else:
    st.write("Handcrafted synthetic example; scoring does not execute a payment.")
    with st.form("transaction"):
        transaction_id = st.text_input("Transaction ID", "demo-1001")
        step = st.number_input("Step", min_value=0, value=42, step=1)
        kind = st.selectbox("Type", ["TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT", "PAYMENT"])
        amount = st.number_input("Amount", min_value=0.0, value=12000.0)
        balance = st.number_input("Sender balance before transaction", min_value=0.0, value=15000.0)
        submitted = st.form_submit_button("Score transaction")
    if submitted:
        result = api(
            "/predict",
            {
                "transaction_id": transaction_id,
                "step": step,
                "type": kind,
                "amount": amount,
                "oldbalanceOrg": balance,
            },
        )
        if result:
            st.metric("Model score / 100", result["risk_score"])
            st.json(result)
            st.caption("Risk bands 30/70 are display defaults, independent of the flag threshold.")
