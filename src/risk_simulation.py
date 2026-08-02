"""
risk_simulation.py
-------------------
Monte Carlo simulation of profit under market and milk-quality variability.

Rather than trusting a single point prediction, this module perturbs the
inputs that are genuinely uncertain at planning time -- market price,
milk quality (fat/SCC-driven quality index), and energy cost -- using
historically observed variability, and re-runs the profit model N times
per candidate product. This produces a profit *distribution* rather than a
single number, which is what a "decision-support model under variability"
should give managers: expected profit, downside risk (5th percentile), and
probability that one product beats another.
"""

import numpy as np
import pandas as pd
from predictive_model import predict_profit


def _historical_std(df: pd.DataFrame, col: str) -> float:
    return float(df[col].std(ddof=1))


def simulate_profit_distribution(
    pipe,
    df_history: pd.DataFrame,
    base_conditions: dict,
    product: str,
    n_sims: int = 5000,
    price_vol_scale: float = 1.0,
    quality_vol_scale: float = 1.0,
    energy_vol_scale: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Perturbs Market_Price_EUR_per_L, milk_quality_index, and
    electricity_cost_per_L using Gaussian noise scaled to historical
    standard deviation (scale factors let you stress-test wider swings than
    observed so far), holding other features at base_conditions.
    """
    rng = np.random.default_rng(seed)

    price_std = _historical_std(df_history, "Market_Price_EUR_per_L") * price_vol_scale
    quality_std = _historical_std(df_history, "milk_quality_index") * quality_vol_scale
    energy_std = _historical_std(df_history, "electricity_cost_per_L") * energy_vol_scale

    profits = np.empty(n_sims)
    for i in range(n_sims):
        row = dict(base_conditions)
        row["Product_Type"] = product
        row["Market_Price_EUR_per_L"] = max(
            0.01, rng.normal(base_conditions["Market_Price_EUR_per_L"], price_std)
        )
        row["milk_quality_index"] = rng.normal(base_conditions["milk_quality_index"], quality_std)
        row["electricity_cost_per_L"] = max(
            0.0, rng.normal(base_conditions["electricity_cost_per_L"], energy_std)
        )
        profits[i] = predict_profit(pipe, row)

    return profits


def summarize_distribution(profits: np.ndarray) -> dict:
    return {
        "mean_EUR": round(float(np.mean(profits)), 2),
        "std_EUR": round(float(np.std(profits)), 2),
        "p5_EUR": round(float(np.percentile(profits, 5)), 2),
        "p95_EUR": round(float(np.percentile(profits, 95)), 2),
        "prob_loss_vs_zero": round(float(np.mean(profits < 0)), 4),
    }


def compare_products_under_risk(pipe, df_history, base_conditions, products, n_sims=5000):
    """Runs the simulation for each candidate product and returns a summary
    table, plus the probability that each product yields the highest profit
    in a given simulated scenario (useful when profit distributions overlap)."""
    all_profits = {p: simulate_profit_distribution(pipe, df_history, base_conditions, p, n_sims) for p in products}

    summary_rows = []
    for p, profits in all_profits.items():
        s = summarize_distribution(profits)
        s["Product_Type"] = p
        summary_rows.append(s)
    summary = pd.DataFrame(summary_rows).set_index("Product_Type")

    stacked = np.vstack([all_profits[p] for p in products])  # shape (n_products, n_sims)
    winner_idx = np.argmax(stacked, axis=0)
    win_prob = {p: round(float(np.mean(winner_idx == i)), 3) for i, p in enumerate(products)}
    summary["Prob_Best_Choice"] = pd.Series(win_prob)

    return summary, all_profits


if __name__ == "__main__":
    from preprocessing import load_processed
    from predictive_model import train
    from decision_engine import build_day_conditions_from_row, PRODUCT_TYPES

    df = load_processed()
    pipe, metrics = train(df)
    conditions = build_day_conditions_from_row(df.iloc[0])

    summary, _ = compare_products_under_risk(pipe, df, conditions, PRODUCT_TYPES)
    print(summary)
