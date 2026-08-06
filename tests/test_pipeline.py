"""
tests/test_pipeline.py
------------------------
Unit tests covering: feature engineering correctness, data validation
(schema + sanity checks), model training/prediction sanity, decision engine
output shape, LP optimization feasibility, and Monte Carlo simulation
output shape.

Run with:  pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from dairy_dss.preprocessing import engineer_features, load_processed
from dairy_dss.data_validation import validate, validate_schema, DataValidationError
from dairy_dss.predictive_model import train, predict_profit
from dairy_dss.decision_engine import recommend_product, build_day_conditions_from_row, PRODUCT_TYPES
from dairy_dss.optimization import estimate_margins_and_shares, optimize_daily_allocation
from dairy_dss.risk_simulation import simulate_profit_distribution
from dairy_dss.config import DATA_PATH


@pytest.fixture(scope="module")
def raw_df():
    return pd.read_csv(DATA_PATH, parse_dates=["Date"])


@pytest.fixture(scope="module")
def processed_df():
    return load_processed()


@pytest.fixture(scope="module")
def trained_pipe(processed_df):
    pipe, metrics = train(processed_df)
    return pipe, metrics


# ---------------------------------------------------------------- schema --

def test_schema_passes_on_real_data(raw_df):
    validate_schema(raw_df)  # should not raise


def test_schema_fails_on_missing_column(raw_df):
    broken = raw_df.drop(columns=["Fat_Percent"])
    with pytest.raises(DataValidationError):
        validate_schema(broken)


def test_value_validation_flags_negative_volume(raw_df):
    broken = raw_df.copy()
    broken.loc[0, "Milk_Collection_L"] = -100
    with pytest.raises(DataValidationError):
        validate(broken, strict=True)


# ------------------------------------------------------- feature engineering --

def test_engineered_features_present(processed_df):
    expected = {
        "milk_quality_index", "capacity_utilization", "storage_utilization",
        "demand_gap", "transport_cost_per_L", "electricity_cost_per_L",
    }
    assert expected.issubset(set(processed_df.columns))


def test_capacity_utilization_is_ratio(processed_df):
    # Should equal Milk_Collection_L / Processing_Capacity_L exactly
    expected = processed_df["Milk_Collection_L"] / processed_df["Processing_Capacity_L"]
    assert np.allclose(processed_df["capacity_utilization"], expected)


def test_no_nulls_after_feature_engineering(processed_df):
    assert processed_df.isnull().sum().sum() == 0


# ------------------------------------------------------------------ model --

def test_model_trains_without_error(trained_pipe):
    pipe, metrics = trained_pipe
    assert pipe is not None
    assert "loocv_mae_eur" in metrics
    assert metrics["loocv_mae_eur"] > 0


def test_model_predictions_are_positive_and_reasonable(trained_pipe, processed_df):
    pipe, _ = trained_pipe
    row = build_day_conditions_from_row(processed_df.iloc[0])
    row["Product_Type"] = "Cheese"
    pred = predict_profit(pipe, row)
    # Predicted profit should be in a plausible range given the historical data
    assert 5000 < pred < 40000


# ------------------------------------------------------------- decision engine --

def test_recommend_product_returns_all_products(trained_pipe, processed_df):
    pipe, _ = trained_pipe
    conditions = build_day_conditions_from_row(processed_df.iloc[0])
    result = recommend_product(pipe, conditions, verbose=False)
    assert set(result["Product_Type"]) == set(PRODUCT_TYPES)
    assert result["Recommended"].sum() == 1  # exactly one top pick


def test_recommendation_ranked_by_profit_descending(trained_pipe, processed_df):
    pipe, _ = trained_pipe
    conditions = build_day_conditions_from_row(processed_df.iloc[0])
    result = recommend_product(pipe, conditions, verbose=False)
    profits = result["Predicted_Profit_EUR"].values
    assert all(profits[i] >= profits[i + 1] for i in range(len(profits) - 1))


# -------------------------------------------------------------- optimization --

def test_lp_allocation_respects_milk_supply_constraint(processed_df):
    margins, shares = estimate_margins_and_shares(processed_df)
    row = processed_df.iloc[0]
    result = optimize_daily_allocation(
        milk_collection_L=row["Milk_Collection_L"],
        processing_capacity_L=row["Processing_Capacity_L"],
        storage_capacity_L=row["Storage_Capacity_L"],
        market_demand_L=row["Market_Demand_L"],
        margins=margins,
        demand_shares=shares,
    )
    assert result["success"]
    total_allocated = sum(result["allocation_L"].values())
    assert total_allocated <= row["Milk_Collection_L"] + 1e-6
    assert total_allocated <= row["Processing_Capacity_L"] + 1e-6


def test_lp_allocation_nonnegative(processed_df):
    margins, shares = estimate_margins_and_shares(processed_df)
    row = processed_df.iloc[0]
    result = optimize_daily_allocation(
        milk_collection_L=row["Milk_Collection_L"],
        processing_capacity_L=row["Processing_Capacity_L"],
        storage_capacity_L=row["Storage_Capacity_L"],
        market_demand_L=row["Market_Demand_L"],
        margins=margins,
        demand_shares=shares,
    )
    for v in result["allocation_L"].values():
        assert v >= -1e-6


# ----------------------------------------------------------- risk simulation --

def test_monte_carlo_output_shape(trained_pipe, processed_df):
    pipe, _ = trained_pipe
    conditions = build_day_conditions_from_row(processed_df.iloc[0])
    conditions["Product_Type"] = "Cheese"
    profits = simulate_profit_distribution(
        pipe, processed_df, conditions, "Cheese", n_sims=200
    )
    assert len(profits) == 200
    assert np.isfinite(profits).all()


def test_monte_carlo_reproducible_with_same_seed(trained_pipe, processed_df):
    pipe, _ = trained_pipe
    conditions = build_day_conditions_from_row(processed_df.iloc[0])
    conditions["Product_Type"] = "Cheese"
    p1 = simulate_profit_distribution(pipe, processed_df, conditions, "Cheese", n_sims=100, seed=1)
    p2 = simulate_profit_distribution(pipe, processed_df, conditions, "Cheese", n_sims=100, seed=1)
    assert np.array_equal(p1, p2)
