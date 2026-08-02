"""
diagnostics.py
----------------
Advanced diagnostic figures for model transparency and robustness review:
    - Correlation heatmap of engineered features
    - LOOCV residual diagnostics (actual vs. predicted, residuals vs. fitted)
    - Model comparison leaderboard chart
    - Bootstrap coefficient forest plot (with confidence intervals)
    - Tornado sensitivity chart (one-at-a-time input perturbation)
    - LP-optimized allocation chart

These are separated from main.py's "headline" figures (profit by product,
recommendation vs. actual, Monte Carlo distribution) because they serve a
different audience: not the operations decision itself, but the analyst/
auditor who wants to check the model is trustworthy.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import FIGURE_DIR, NUMERIC_FEATURES, TARGET
from model_selection import build_pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict

logger = logging.getLogger(__name__)


def plot_correlation_heatmap(df: pd.DataFrame):
    cols = NUMERIC_FEATURES + [TARGET]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close(fig)


def plot_model_comparison(leaderboard: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(leaderboard["model"], leaderboard["loocv_mae_eur"], color="#4472C4")
    axes[0].set_ylabel("LOOCV MAE (EUR, lower is better)")
    axes[0].set_title("Model Comparison — Error")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(leaderboard["model"], leaderboard["loocv_r2"], color="#3B7A57")
    axes[1].set_ylabel("LOOCV R² (higher is better)")
    axes[1].set_title("Model Comparison — Fit")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].axhline(0, color="grey", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "model_comparison.png"), dpi=150)
    plt.close(fig)


def plot_loocv_residuals(df: pd.DataFrame, model_name: str = "Ridge"):
    from config import CATEGORICAL_FEATURES
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    pipe = build_pipeline(model_name)
    y_pred = cross_val_predict(pipe, X, y, cv=LeaveOneOut())
    residuals = y.values - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    lims = [min(y.min(), y_pred.min()) - 300, max(y.max(), y_pred.max()) + 300]
    axes[0].scatter(y, y_pred, color="#4472C4", s=60)
    axes[0].plot(lims, lims, linestyle="--", color="grey", label="Perfect prediction")
    axes[0].set_xlabel("Actual Profit (EUR)")
    axes[0].set_ylabel("LOOCV-Predicted Profit (EUR)")
    axes[0].set_title(f"{model_name}: Actual vs. Predicted (LOOCV)")
    axes[0].legend()

    axes[1].scatter(y_pred, residuals, color="#C9A227", s=60)
    axes[1].axhline(0, color="grey", linestyle="--")
    axes[1].set_xlabel("LOOCV-Predicted Profit (EUR)")
    axes[1].set_ylabel("Residual (Actual − Predicted, EUR)")
    axes[1].set_title("Residuals vs. Fitted")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "loocv_residuals.png"), dpi=150)
    plt.close(fig)


def plot_bootstrap_coefficients(boot_summary: pd.DataFrame):
    boot_summary = boot_summary.sort_values("point_estimate")
    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(boot_summary))

    for i, (_, row) in enumerate(boot_summary.iterrows()):
        color = "#3B7A57" if row["significant_at_90%"] else "#B0B0B0"
        xerr_low = row["boot_mean"] - row["ci_low_5%"]
        xerr_high = row["ci_high_95%"] - row["boot_mean"]
        ax.errorbar(
            row["boot_mean"], i,
            xerr=[[xerr_low], [xerr_high]],
            fmt="o", color="black", ecolor=color, elinewidth=4, capsize=3,
        )
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(boot_summary["feature"].str.replace("num__", "").str.replace("cat__", ""))
    ax.set_xlabel("Coefficient value (EUR impact on profit, standardised features)")
    ax.set_title("Bootstrap Coefficient Stability (90% CI)\nGreen = reliably non-zero at n=10; Grey = not distinguishable from noise")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "bootstrap_coefficients.png"), dpi=150)
    plt.close(fig)


def tornado_sensitivity(pipe, base_conditions: dict, pct_swing: float = 0.15) -> pd.DataFrame:
    """
    One-at-a-time sensitivity analysis: perturbs each numeric feature by
    +/- pct_swing (default 15%) from its base value, holding all others
    fixed, and records the resulting change in predicted profit. Useful to
    see which inputs the recommendation is most sensitive to.
    """
    from predictive_model import predict_profit

    base_profit = predict_profit(pipe, base_conditions)
    rows = []
    for feat in NUMERIC_FEATURES:
        base_val = base_conditions[feat]
        low = dict(base_conditions); low[feat] = base_val * (1 - pct_swing)
        high = dict(base_conditions); high[feat] = base_val * (1 + pct_swing)
        p_low = predict_profit(pipe, low)
        p_high = predict_profit(pipe, high)
        rows.append({
            "feature": feat,
            "profit_at_low": p_low,
            "profit_at_high": p_high,
            "swing_eur": abs(p_high - p_low),
        })
    result = pd.DataFrame(rows).sort_values("swing_eur", ascending=True)
    result.attrs["base_profit"] = base_profit
    return result


def plot_tornado(tornado_df: pd.DataFrame):
    base = tornado_df.attrs.get("base_profit", 0)
    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(tornado_df))

    lows = tornado_df["profit_at_low"] - base
    highs = tornado_df["profit_at_high"] - base

    for i, (lo, hi) in enumerate(zip(lows, highs)):
        left = min(lo, hi)
        width = abs(hi - lo)
        color = "#C9527A" if hi < lo else "#4472C4"
        ax.barh(i, width, left=left, color=color, alpha=0.8)

    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tornado_df["feature"])
    ax.set_xlabel(f"Change in predicted profit vs. base (EUR)\nBase prediction: €{base:,.0f}")
    ax.set_title("Sensitivity Analysis: ±15% Input Perturbation (Tornado Chart)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "tornado_sensitivity.png"), dpi=150)
    plt.close(fig)


def plot_lp_allocation(allocation_result: dict, title_suffix: str = ""):
    allocation = allocation_result["allocation_L"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    products = list(allocation.keys())
    values = list(allocation.values())
    colors = ["#3B7A57", "#C9A227", "#4472C4"]
    bars = ax.bar(products, values, color=colors[: len(products)])
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(values) * 0.01, f"{v:,.0f} L",
                ha="center", fontsize=9)
    ax.set_ylabel("Allocated Milk Volume (L)")
    ax.set_title(f"LP-Optimal Multi-Product Allocation {title_suffix}\n"
                 f"Total predicted profit: €{allocation_result['total_predicted_profit_EUR']:,.0f}"
                 f" | Unallocated: {allocation_result['milk_unallocated_L']:,.0f} L")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "lp_optimal_allocation.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import os as _os
    from config import ensure_output_dirs
    from preprocessing import load_processed
    from model_selection import select_best_model, bootstrap_ridge_coefficients
    from decision_engine import build_day_conditions_from_row
    from optimization import estimate_margins_and_shares, optimize_from_row

    ensure_output_dirs()
    df = load_processed()

    plot_correlation_heatmap(df)

    pipe, best_name, leaderboard = select_best_model(df)
    plot_model_comparison(leaderboard)
    plot_loocv_residuals(df, best_name)

    boot = bootstrap_ridge_coefficients(df)
    plot_bootstrap_coefficients(boot)

    conditions = build_day_conditions_from_row(df.iloc[-1])
    conditions["Product_Type"] = df.iloc[-1]["Product_Type"]
    tornado = tornado_sensitivity(pipe, conditions)
    plot_tornado(tornado)

    margins, shares = estimate_margins_and_shares(df)
    lp_result = optimize_from_row(df.iloc[-1], margins, shares)
    if lp_result["success"]:
        plot_lp_allocation(lp_result, title_suffix="(latest day)")

    print("All diagnostic figures written to", FIGURE_DIR)
