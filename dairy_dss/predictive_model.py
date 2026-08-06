"""
predictive_model.py
--------------------
Thin wrapper around the selected best model (see model_selection.py) for
querying profit predictions on hypothetical inputs -- e.g. "what would
profit be if this milk were processed into Butter instead of Cheese?".

Model selection, LOOCV evaluation, and bootstrap coefficient stability all
live in model_selection.py; this module exposes the simple predict/train
interface used by decision_engine.py and risk_simulation.py.
"""

import logging
import pandas as pd

from .config import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from .model_selection import select_best_model

logger = logging.getLogger(__name__)


def train(df: pd.DataFrame = None):
    """Returns (fitted_pipeline, metrics_dict). Selects the best-performing
    model out of the LOOCV-compared candidates in model_selection.py."""
    if df is None:
        from .preprocessing import load_processed
        df = load_processed()
    pipe, best_name, leaderboard = select_best_model(df)
    best_row = leaderboard.iloc[0]
    metrics = {
        "selected_model": best_name,
        "loocv_mae_eur": best_row["loocv_mae_eur"],
        "loocv_rmse_eur": best_row["loocv_rmse_eur"],
        "loocv_r2": best_row["loocv_r2"],
    }
    return pipe, metrics


def predict_profit(pipe, row: dict) -> float:
    """
    row: dict containing all NUMERIC_FEATURES + CATEGORICAL_FEATURES keys.
    Returns predicted profit in EUR.
    """
    X = pd.DataFrame([row])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    return float(pipe.predict(X)[0])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from .preprocessing import load_processed
    df = load_processed()
    pipe, metrics = train(df)
    print("Model performance (Leave-One-Out CV):", metrics)
