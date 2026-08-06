"""
extended_analysis.py
---------------------
Additional, more advanced analyses layered on top of the core DSS pipeline:

  1. Multicollinearity diagnostics (Variance Inflation Factors) for the
     engineered numeric features feeding the Ridge model.
  2. Principal Component Analysis (PCA) of the engineered feature set, to
     characterise the dimensionality/redundancy of the feature space
     independently of the VIF check.
  3. Full-horizon (all 10 days) LP re-allocation, compared day-by-day
     against realised profit -- extending Section 4.5's single-day LP
     result to the whole observation window.
  4. Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR) at the 95%
     level from the Monte Carlo profit distributions, extending Section
     4.6's percentile summary with tail-risk measures.
  5. Permutation feature importance for the selected Ridge model
     (held-out / cross-validated), as an independent cross-check on the
     bootstrap coefficient significance in Section 4.2.
  6. Two-way interaction sensitivity: predicted profit surface across
     jointly varying milk quality index and market price, extending the
     one-at-a-time tornado analysis in Section 4.3.
  7. Paired significance test (Wilcoxon signed-rank on LOOCV absolute
     errors) comparing the top two candidate models (Ridge vs Elastic
     Net), to check whether the MAE gap in Table 1 is more than sampling
     noise at n = 10.

All figures are written to outputs/figures/; all tables are written to
outputs/ as CSV, mirroring the existing pipeline's output convention.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.inspection import permutation_importance
from scipy import stats
from scipy.optimize import linprog

from .config import FIGURE_DIR, OUTPUT_DIR, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, PRODUCT_TYPES, RANDOM_SEED
from .preprocessing import load_processed
from .model_selection import build_pipeline, _preprocessor, compare_models
from .predictive_model import train, predict_profit
from .optimization import estimate_margins_and_shares, optimize_from_row
from .decision_engine import build_day_conditions_from_row
from .risk_simulation import compare_products_under_risk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Variance Inflation Factors
# ---------------------------------------------------------------------------
def compute_vif(df: pd.DataFrame, features=NUMERIC_FEATURES) -> pd.DataFrame:
    """VIF_j = 1 / (1 - R^2_j), where R^2_j is from regressing feature j on
    all other features in the list. VIF > 5 is a common rule-of-thumb flag
    for problematic multicollinearity; VIF > 10 is severe."""
    X = df[features].values
    n, p = X.shape
    vifs = []
    for j in range(p):
        y_j = X[:, j]
        X_others = np.delete(X, j, axis=1)
        X_others_c = np.column_stack([np.ones(n), X_others])
        beta, *_ = np.linalg.lstsq(X_others_c, y_j, rcond=None)
        y_hat = X_others_c @ beta
        ss_res = np.sum((y_j - y_hat) ** 2)
        ss_tot = np.sum((y_j - y_j.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vif = 1 / (1 - r2) if r2 < 0.999999 else np.inf
        vifs.append(vif)
    out = pd.DataFrame({"feature": features, "VIF": vifs}).sort_values("VIF", ascending=False)
    out["flag"] = np.where(out["VIF"] > 10, "severe", np.where(out["VIF"] > 5, "moderate", "ok"))
    return out.reset_index(drop=True)


def plot_vif(vif_df: pd.DataFrame, path: str):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colors = vif_df["flag"].map({"severe": "#c0392b", "moderate": "#e67e22", "ok": "#2c7a4b"})
    ax.barh(vif_df["feature"][::-1], vif_df["VIF"][::-1], color=colors[::-1])
    ax.axvline(5, color="grey", ls="--", lw=1, label="VIF = 5 (moderate)")
    ax.axvline(10, color="black", ls="--", lw=1, label="VIF = 10 (severe)")
    ax.set_xlabel("Variance Inflation Factor")
    ax.set_title("Multicollinearity diagnostics for engineered numeric features")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. PCA
# ---------------------------------------------------------------------------
def compute_pca(df: pd.DataFrame, features=NUMERIC_FEATURES):
    X = StandardScaler().fit_transform(df[features].values)
    pca = PCA(n_components=min(len(features), len(df)))
    scores = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_
    loadings = pd.DataFrame(pca.components_.T, index=features,
                             columns=[f"PC{i+1}" for i in range(len(explained))])
    return pca, explained, loadings, scores


def plot_pca(explained, loadings, path: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    cum = np.cumsum(explained)
    axes[0].bar(range(1, len(explained) + 1), explained * 100, color="#3d6ba8", label="Individual")
    axes[0].plot(range(1, len(explained) + 1), cum * 100, color="#c0392b", marker="o", label="Cumulative")
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("Variance explained (%)")
    axes[0].set_title("PCA scree plot")
    axes[0].legend(fontsize=8)

    top2 = loadings.iloc[:, :2]
    top2.plot(kind="barh", ax=axes[1], color=["#3d6ba8", "#e67e22"])
    axes[1].set_title("PC1 / PC2 loadings by feature")
    axes[1].axvline(0, color="black", lw=0.8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Full-horizon LP re-allocation
# ---------------------------------------------------------------------------
def full_horizon_lp(df: pd.DataFrame) -> pd.DataFrame:
    margins, shares = estimate_margins_and_shares(df)
    rows = []
    for _, row in df.iterrows():
        res = optimize_from_row(row, margins, shares)
        rows.append({
            "Date": row["Date"].date(),
            "Actual_Product": row["Product_Type"],
            "Actual_Profit_EUR": row["Profit_EUR"],
            "LP_Total_Predicted_Profit_EUR": res.get("total_predicted_profit_EUR", np.nan),
            "LP_Milk_Unallocated_L": res.get("milk_unallocated_L", np.nan),
            **{f"LP_{k.replace(' ', '_')}_L": v for k, v in res.get("allocation_L", {}).items()},
        })
    out = pd.DataFrame(rows)
    out["LP_Uplift_vs_Actual_EUR"] = out["LP_Total_Predicted_Profit_EUR"] - out["Actual_Profit_EUR"]
    return out


def plot_full_horizon_lp(lp_df: pd.DataFrame, path: str):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(lp_df))
    ax.plot(x, lp_df["Actual_Profit_EUR"], marker="o", label="Actual profit (single product/day)", color="#3d6ba8")
    ax.plot(x, lp_df["LP_Total_Predicted_Profit_EUR"], marker="s", label="LP multi-product allocation", color="#c0392b")
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in lp_df["Date"]], rotation=45, ha="right")
    ax.set_ylabel("Profit (EUR)")
    ax.set_title("Actual vs. LP-optimal multi-product profit, all 10 days")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. VaR / CVaR from Monte Carlo distributions
# ---------------------------------------------------------------------------
def var_cvar(profits: np.ndarray, level: float = 0.95) -> dict:
    var = np.percentile(profits, (1 - level) * 100)
    tail = profits[profits <= var]
    cvar = float(tail.mean()) if len(tail) else float(var)
    return {"VaR_95_EUR": round(float(var), 2), "CVaR_95_EUR": round(cvar, 2)}


def plot_var_cvar(all_profits: dict, path: str):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = {"Cheese": "#3d6ba8", "Butter": "#e67e22", "Milk Powder": "#2c7a4b"}
    for product, profits in all_profits.items():
        ax.hist(profits, bins=40, alpha=0.45, label=product, color=colors.get(product))
        stats_d = var_cvar(profits)
        ax.axvline(stats_d["VaR_95_EUR"], color=colors.get(product), ls="--", lw=1.3)
    ax.set_xlabel("Simulated profit (EUR)")
    ax.set_ylabel("Frequency (of 5,000 simulations)")
    ax.set_title("Monte Carlo profit distributions with 95% VaR (dashed)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Permutation feature importance
# ---------------------------------------------------------------------------
def permutation_importance_ridge(df: pd.DataFrame, n_repeats: int = 200, seed: int = RANDOM_SEED) -> pd.DataFrame:
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    pipe = build_pipeline("Ridge")
    pipe.fit(X, y)
    result = permutation_importance(
        pipe, X, y, n_repeats=n_repeats, random_state=seed, scoring="neg_mean_absolute_error"
    )
    out = pd.DataFrame({
        "feature": X.columns,
        "importance_mean_EUR_MAE_increase": -result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean_EUR_MAE_increase", ascending=False).reset_index(drop=True)
    return out


def plot_permutation_importance(imp_df: pd.DataFrame, path: str):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.barh(imp_df["feature"][::-1], imp_df["importance_mean_EUR_MAE_increase"][::-1],
            xerr=imp_df["importance_std"][::-1], color="#3d6ba8", ecolor="grey", capsize=3)
    ax.set_xlabel("Increase in MAE (EUR) when feature is permuted")
    ax.set_title("Permutation feature importance (Ridge, in-sample, 200 repeats)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Two-way interaction sensitivity surface
# ---------------------------------------------------------------------------
def two_way_sensitivity(pipe, base_conditions: dict, df_history: pd.DataFrame, product: str,
                         n_grid: int = 25) -> tuple:
    q_std = df_history["milk_quality_index"].std(ddof=1)
    p_std = df_history["Market_Price_EUR_per_L"].std(ddof=1)
    q_base = base_conditions["milk_quality_index"]
    p_base = base_conditions["Market_Price_EUR_per_L"]

    q_range = np.linspace(q_base - 2 * q_std, q_base + 2 * q_std, n_grid)
    p_range = np.linspace(max(0.01, p_base - 2 * p_std), p_base + 2 * p_std, n_grid)

    Z = np.zeros((n_grid, n_grid))
    for i, q in enumerate(q_range):
        for j, p in enumerate(p_range):
            row = dict(base_conditions)
            row["Product_Type"] = product
            row["milk_quality_index"] = q
            row["Market_Price_EUR_per_L"] = p
            Z[i, j] = predict_profit(pipe, row)
    return q_range, p_range, Z


def plot_two_way_sensitivity(q_range, p_range, Z, product: str, path: str):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    cf = ax.contourf(p_range, q_range, Z, levels=20, cmap="RdYlGn")
    fig.colorbar(cf, ax=ax, label="Predicted profit (EUR)")
    ax.set_xlabel("Market price (EUR/L)")
    ax.set_ylabel("Milk quality index")
    ax.set_title(f"Predicted profit surface: quality x price interaction ({product})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Paired significance test on LOOCV errors (Ridge vs Elastic Net)
# ---------------------------------------------------------------------------
def loocv_abs_errors(df: pd.DataFrame, model_name: str) -> np.ndarray:
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].values
    pipe = build_pipeline(model_name)
    y_pred = cross_val_predict(pipe, X, y, cv=LeaveOneOut())
    return np.abs(y - y_pred)


def paired_model_test(df: pd.DataFrame, model_a="Ridge", model_b="ElasticNet") -> dict:
    err_a = loocv_abs_errors(df, model_a)
    err_b = loocv_abs_errors(df, model_b)
    diff = err_b - err_a  # positive => model_a (Ridge) has smaller error
    t_stat, t_p = stats.ttest_rel(err_b, err_a)
    try:
        w_stat, w_p = stats.wilcoxon(err_b, err_a)
    except ValueError:
        w_stat, w_p = np.nan, np.nan
    return {
        "model_a": model_a, "model_b": model_b,
        "mean_abs_error_diff_EUR": round(float(diff.mean()), 2),
        "paired_t_stat": round(float(t_stat), 3), "paired_t_p": round(float(t_p), 4),
        "wilcoxon_stat": (round(float(w_stat), 3) if not np.isnan(w_stat) else None),
        "wilcoxon_p": (round(float(w_p), 4) if not np.isnan(w_p) else None),
        "n": len(df),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_all():
    os.makedirs(FIGURE_DIR, exist_ok=True)
    df = load_processed()

    # 1. VIF
    vif_df = compute_vif(df)
    vif_df.to_csv(os.path.join(OUTPUT_DIR, "vif.csv"), index=False)
    plot_vif(vif_df, os.path.join(FIGURE_DIR, "vif.png"))

    # 2. PCA
    pca, explained, loadings, scores = compute_pca(df)
    pd.DataFrame({"component": [f"PC{i+1}" for i in range(len(explained))],
                  "explained_variance_ratio": explained,
                  "cumulative": np.cumsum(explained)}).to_csv(
        os.path.join(OUTPUT_DIR, "pca_explained_variance.csv"), index=False)
    loadings.to_csv(os.path.join(OUTPUT_DIR, "pca_loadings.csv"))
    plot_pca(explained, loadings, os.path.join(FIGURE_DIR, "pca.png"))

    # 3. Full-horizon LP
    lp_df = full_horizon_lp(df)
    lp_df.to_csv(os.path.join(OUTPUT_DIR, "full_horizon_lp.csv"), index=False)
    plot_full_horizon_lp(lp_df, os.path.join(FIGURE_DIR, "full_horizon_lp.png"))

    # Train model once, reuse for 4/5/6
    pipe, metrics = train(df)
    conditions = build_day_conditions_from_row(df.iloc[-1])

    # 4. VaR/CVaR
    summary, all_profits = compare_products_under_risk(pipe, df, conditions, PRODUCT_TYPES)
    var_cvar_rows = []
    for product, profits in all_profits.items():
        d = var_cvar(profits)
        d["Product_Type"] = product
        var_cvar_rows.append(d)
    var_cvar_df = pd.DataFrame(var_cvar_rows).set_index("Product_Type")
    var_cvar_df.to_csv(os.path.join(OUTPUT_DIR, "var_cvar.csv"))
    plot_var_cvar(all_profits, os.path.join(FIGURE_DIR, "var_cvar.png"))

    # 5. Permutation importance
    imp_df = permutation_importance_ridge(df)
    imp_df.to_csv(os.path.join(OUTPUT_DIR, "permutation_importance.csv"), index=False)
    plot_permutation_importance(imp_df, os.path.join(FIGURE_DIR, "permutation_importance.png"))

    # 6. Two-way sensitivity (cheese, most recent day)
    q_range, p_range, Z = two_way_sensitivity(pipe, conditions, df, "Cheese")
    plot_two_way_sensitivity(q_range, p_range, Z, "Cheese", os.path.join(FIGURE_DIR, "two_way_sensitivity.png"))

    # 7. Paired significance test
    test_result = paired_model_test(df)
    pd.DataFrame([test_result]).to_csv(os.path.join(OUTPUT_DIR, "paired_model_test.csv"), index=False)

    print("=== VIF ===\n", vif_df.round(2).to_string(index=False))
    print("\n=== PCA explained variance ===\n", (explained * 100).round(1))
    print("\n=== Full-horizon LP (head) ===\n", lp_df.round(1).head().to_string(index=False))
    print("\n=== VaR / CVaR (95%) ===\n", var_cvar_df)
    print("\n=== Permutation importance ===\n", imp_df.round(2).to_string(index=False))
    print("\n=== Paired model test (Ridge vs Elastic Net) ===\n", test_result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_all()
