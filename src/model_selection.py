"""
model_selection.py
--------------------
Compares multiple candidate regression algorithms for the profit-prediction
task using Leave-One-Out Cross-Validation, and quantifies coefficient
uncertainty for the selected model via bootstrap resampling.

Why compare models at all with n=10? Because picking a single algorithm
up front (e.g. "just use Ridge") is itself an assumption worth checking.
Comparing a regularised linear model against a couple of non-linear/
tree-based learners on the SAME cross-validation protocol makes that
assumption explicit and auditable, rather than hidden.

Why bootstrap the coefficients? With 10 observations, a single Ridge fit's
coefficients are a single point estimate with no sense of how stable they
are. Bootstrap resampling (refitting on resampled-with-replacement subsets)
gives a confidence interval for each coefficient -- if an interval crosses
zero, that feature's effect is not reliably distinguishable from noise at
this sample size, and the decision engine's explanations should say so.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, RANDOM_SEED, N_BOOTSTRAP

logger = logging.getLogger(__name__)


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


CANDIDATE_MODELS = {
    "Ridge": lambda: Ridge(alpha=2.0),
    "ElasticNet": lambda: ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=5000),
    "RandomForest": lambda: RandomForestRegressor(
        n_estimators=300, max_depth=3, random_state=RANDOM_SEED
    ),
    "GradientBoosting": lambda: GradientBoostingRegressor(
        n_estimators=150, max_depth=2, learning_rate=0.05, random_state=RANDOM_SEED
    ),
}


def build_pipeline(model_name: str) -> Pipeline:
    if model_name not in CANDIDATE_MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Options: {list(CANDIDATE_MODELS)}")
    return Pipeline([("pre", _preprocessor()), ("model", CANDIDATE_MODELS[model_name]())])


def compare_models(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluates every candidate model with LOOCV and returns a leaderboard
    sorted by MAE (ascending, i.e. best first)."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    loo = LeaveOneOut()

    rows = []
    for name in CANDIDATE_MODELS:
        pipe = build_pipeline(name)
        y_pred = cross_val_predict(pipe, X, y, cv=loo)
        rows.append({
            "model": name,
            "loocv_mae_eur": round(mean_absolute_error(y, y_pred), 2),
            "loocv_rmse_eur": round(mean_squared_error(y, y_pred) ** 0.5, 2),
            "loocv_r2": round(r2_score(y, y_pred), 3),
        })
        logger.info("Evaluated %s: MAE=%.2f R2=%.3f", name, rows[-1]["loocv_mae_eur"], rows[-1]["loocv_r2"])

    leaderboard = pd.DataFrame(rows).sort_values("loocv_mae_eur").reset_index(drop=True)
    return leaderboard


def select_best_model(df: pd.DataFrame):
    """Returns (fitted_pipeline, model_name, leaderboard_df)."""
    leaderboard = compare_models(df)
    best_name = leaderboard.iloc[0]["model"]
    logger.info("Selected best model: %s", best_name)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    pipe = build_pipeline(best_name)
    pipe.fit(X, y)
    return pipe, best_name, leaderboard


def bootstrap_ridge_coefficients(df: pd.DataFrame, n_boot: int = N_BOOTSTRAP, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Bootstrap confidence intervals for the Ridge model's coefficients (the
    most interpretable candidate). Resamples rows with replacement, refits,
    and records each coefficient's distribution across resamples.
    """
    rng = np.random.default_rng(seed)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].values
    n = len(df)

    # Fit the preprocessor ONCE on the full data and reuse it for every
    # bootstrap resample. Refitting the OneHotEncoder per-resample is unsafe
    # here: a resample can by chance omit a product category entirely
    # (n=10), which would silently change the feature dimensionality.
    # Only the Ridge coefficients are refit per resample -- which is the
    # actual quantity we want a confidence interval for.
    preprocessor = _preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()

    pipe0 = Ridge(alpha=2.0)
    pipe0.fit(X_transformed, y)

    coef_samples = np.zeros((n_boot, len(feature_names)))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        model_b = Ridge(alpha=2.0)
        model_b.fit(X_transformed[idx], y[idx])
        coef_samples[b, :] = model_b.coef_

    summary = pd.DataFrame({
        "feature": feature_names,
        "point_estimate": pipe0.coef_,
        "boot_mean": coef_samples.mean(axis=0),
        "ci_low_5%": np.percentile(coef_samples, 5, axis=0),
        "ci_high_95%": np.percentile(coef_samples, 95, axis=0),
    })
    summary["significant_at_90%"] = np.sign(summary["ci_low_5%"]) == np.sign(summary["ci_high_95%"])
    return summary.sort_values("point_estimate", key=abs, ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from preprocessing import load_processed

    data = load_processed()
    board = compare_models(data)
    print("=== Model comparison (LOOCV) ===")
    print(board.to_string(index=False))

    print("\n=== Bootstrap coefficient stability (Ridge, 2000 resamples) ===")
    boot = bootstrap_ridge_coefficients(data)
    print(boot.round(2).to_string(index=False))
