"""
main.py
-------
CLI entry point for the Integrated Decision-Support Model. Runs the full
pipeline end to end:

  1. Load, validate, and feature-engineer the data
  2. Compare candidate models (LOOCV) and select the best
  3. Quantify coefficient uncertainty via bootstrap resampling
  4. Generate day-by-day product recommendations (model vs. actual)
  5. Run Monte Carlo risk simulation for a chosen day
  6. Solve the LP capacity-constrained multi-product allocation problem
  7. Generate the full figure set (headline + diagnostic)

Usage:
    python main.py                          # run with defaults
    python main.py --n-sims 10000            # more Monte Carlo simulations
    python main.py --data ../data/other.csv  # point at a different dataset
    python main.py --log-level DEBUG
"""

import argparse
import logging
import sys

import pandas as pd

from config import (
    DATA_PATH, OUTPUT_DIR, FIGURE_DIR, N_MONTE_CARLO_SIMS,
    PRODUCT_TYPES, ensure_output_dirs,
)
from preprocessing import load_processed
from model_selection import select_best_model, bootstrap_ridge_coefficients, compare_models
from decision_engine import recommend_product, build_day_conditions_from_row
from risk_simulation import compare_products_under_risk
from optimization import estimate_margins_and_shares, optimize_from_row
import diagnostics

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Dairy Processing Decision-Support Model")
    p.add_argument("--data", default=DATA_PATH, help="Path to input CSV")
    p.add_argument("--n-sims", type=int, default=N_MONTE_CARLO_SIMS, help="Monte Carlo simulations per product")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def run_all_recommendations(df: pd.DataFrame, pipe) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        conditions = build_day_conditions_from_row(row)
        rec = recommend_product(pipe, conditions, verbose=False)
        best = rec.iloc[0]
        records.append({
            "Date": row["Date"].date(),
            "Actual_Product": row["Product_Type"],
            "Actual_Profit_EUR": row["Profit_EUR"],
            "Model_Recommended_Product": best["Product_Type"],
            "Model_Predicted_Profit_EUR": best["Predicted_Profit_EUR"],
            "Matches_Actual_Choice": best["Product_Type"] == row["Product_Type"],
        })
    return pd.DataFrame(records)


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s:%(name)s:%(message)s")
    ensure_output_dirs()

    logger.info("=== Step 1: Load & validate data ===")
    df = load_processed(args.data)

    logger.info("=== Step 2: Model comparison (LOOCV) ===")
    pipe, best_name, leaderboard = select_best_model(df)
    print("\nModel leaderboard (LOOCV):")
    print(leaderboard.to_string(index=False))
    print(f"\nSelected model: {best_name}")

    logger.info("=== Step 3: Bootstrap coefficient stability ===")
    boot_summary = bootstrap_ridge_coefficients(df)
    boot_summary.to_csv(f"{OUTPUT_DIR}/bootstrap_coefficients.csv", index=False)
    print("\nBootstrap coefficient stability (90% CI, Ridge):")
    print(boot_summary.round(2).to_string(index=False))

    logger.info("=== Step 4: Day-by-day recommendations ===")
    results_df = run_all_recommendations(df, pipe)
    results_df.to_csv(f"{OUTPUT_DIR}/recommendations_vs_actual.csv", index=False)
    print("\nModel recommendation vs. actual historical choice:")
    print(results_df.to_string(index=False))

    logger.info("=== Step 5: Monte Carlo risk simulation (latest day) ===")
    latest_conditions = build_day_conditions_from_row(df.iloc[-1])
    risk_summary, all_profits = compare_products_under_risk(
        pipe, df, latest_conditions, PRODUCT_TYPES, n_sims=args.n_sims
    )
    risk_summary.to_csv(f"{OUTPUT_DIR}/risk_summary_latest_day.csv")
    print(f"\nMonte Carlo risk summary (latest day, {args.n_sims} sims):")
    print(risk_summary)

    logger.info("=== Step 6: LP capacity-constrained allocation (latest day) ===")
    margins, shares = estimate_margins_and_shares(df)
    lp_result = optimize_from_row(df.iloc[-1], margins, shares)
    print("\nLP-optimal multi-product allocation (latest day):")
    print(lp_result)

    logger.info("=== Step 7: Generate figures ===")
    diagnostics.plot_correlation_heatmap(df)
    diagnostics.plot_model_comparison(leaderboard)
    diagnostics.plot_loocv_residuals(df, best_name)
    diagnostics.plot_bootstrap_coefficients(boot_summary)

    tornado_conditions = dict(latest_conditions)
    tornado_conditions["Product_Type"] = df.iloc[-1]["Product_Type"]
    tornado = diagnostics.tornado_sensitivity(pipe, tornado_conditions)
    diagnostics.plot_tornado(tornado)

    if lp_result["success"]:
        diagnostics.plot_lp_allocation(lp_result, title_suffix="(latest day)")

    _plot_headline_figures(df, results_df, all_profits)

    print(f"\nAll figures written to {FIGURE_DIR}/")
    print(f"All tables written to {OUTPUT_DIR}/")


def _plot_headline_figures(df, results_df, all_profits):
    """Operational (non-diagnostic) figures: profit by product, recommendation
    vs. actual over time, Monte Carlo distribution, quality vs. profit."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    means = df.groupby("Product_Type")["Profit_EUR"].mean().reindex(PRODUCT_TYPES)
    stds = df.groupby("Product_Type")["Profit_EUR"].std().reindex(PRODUCT_TYPES)
    ax.bar(means.index, means.values, yerr=stds.values, capsize=6, color=["#3B7A57", "#C9A227", "#4472C4"])
    ax.set_ylabel("Mean Profit (EUR)")
    ax.set_title("Historical Profit by Product Type (± 1 SD)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "profit_by_product.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(results_df))
    ax.plot(x, results_df["Actual_Profit_EUR"], marker="o", label="Actual Profit")
    ax.plot(x, results_df["Model_Predicted_Profit_EUR"], marker="s", linestyle="--",
            label="Model-Recommended Choice: Predicted Profit")
    ax.set_xticks(list(x))
    ax.set_xticklabels([d.strftime("%m-%d") for d in pd.to_datetime(results_df["Date"])], rotation=45)
    ax.set_ylabel("Profit (EUR)")
    ax.set_title("Actual vs. Model-Recommended Profit by Day")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "recommendation_vs_actual.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for product, profits in all_profits.items():
        ax.hist(profits, bins=40, alpha=0.55, label=product, density=True)
    ax.set_xlabel("Simulated Profit (EUR)")
    ax.set_ylabel("Density")
    ax.set_title("Monte Carlo Profit Distribution by Product\n(under market price / milk quality / energy cost variability)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "risk_distribution.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"Cheese": "#3B7A57", "Butter": "#C9A227", "Milk Powder": "#4472C4"}
    for product in PRODUCT_TYPES:
        sub = df[df["Product_Type"] == product]
        ax.scatter(sub["milk_quality_index"], sub["Profit_EUR"], label=product, color=colors[product], s=70)
    ax.set_xlabel("Milk Quality Index")
    ax.set_ylabel("Profit (EUR)")
    ax.set_title("Milk Quality Index vs. Profit, by Product")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "quality_vs_profit.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
