"""
decision_engine.py
-------------------
Core decision-support logic: given a day's milk-quality and market
conditions, evaluate the predicted profit under each candidate product
(Cheese, Butter, Milk Powder) and recommend the most profitable choice,
subject to processing-capacity and market-demand feasibility checks.
"""

import logging
import pandas as pd

from config import NUMERIC_FEATURES, PRODUCT_TYPES
from predictive_model import predict_profit

logger = logging.getLogger(__name__)


def recommend_product(pipe, day_conditions: dict, verbose: bool = True) -> pd.DataFrame:
    """
    day_conditions: dict with keys matching NUMERIC_FEATURES (Product_Type
    is swapped internally for each candidate).

    Returns a DataFrame ranked by predicted profit, with a feasibility flag
    based on Processing_Capacity_L / Market_Demand_L (if supplied in
    day_conditions under 'Processing_Capacity_L' / 'Market_Demand_L').
    """
    rows = []
    for product in PRODUCT_TYPES:
        row = dict(day_conditions)
        row["Product_Type"] = product
        predicted_profit = predict_profit(pipe, row)
        rows.append({"Product_Type": product, "Predicted_Profit_EUR": round(predicted_profit, 2)})

    result = pd.DataFrame(rows).sort_values("Predicted_Profit_EUR", ascending=False).reset_index(drop=True)
    result["Rank"] = result.index + 1
    result["Recommended"] = result["Rank"] == 1

    if verbose:
        print("Decision recommendation for given day conditions:")
        print(result.to_string(index=False))

    return result


def build_day_conditions_from_row(row: pd.Series) -> dict:
    """Helper: build a day_conditions dict from a row of the processed dataframe
    (useful for re-evaluating historical days 'what if a different product had
    been chosen')."""
    return {feat: row[feat] for feat in NUMERIC_FEATURES}


if __name__ == "__main__":
    from preprocessing import load_processed
    from predictive_model import train

    df = load_processed()
    pipe, metrics = train(df)
    print("Model metrics:", metrics)
    print()

    # Example: re-evaluate the first day under all three product options
    example_conditions = build_day_conditions_from_row(df.iloc[0])
    recommend_product(pipe, example_conditions)
