"""
optimization.py
-----------------
Linear-programming layer for capacity-constrained, multi-product daily
production planning.

The predictive model answers "which single product is best for today?".
This module answers a different, complementary question: "if the plant
COULD split today's milk across multiple product lines simultaneously,
how should it allocate litres across Cheese / Butter / Milk Powder to
maximise total profit, subject to processing capacity, storage capacity,
and market demand?"

Formulation (per day):
    maximise   sum_i ( margin_i * x_i )
    subject to sum_i x_i          <= Milk_Collection_L      (can't process more milk than collected)
               sum_i x_i          <= Processing_Capacity_L  (plant throughput limit)
               sum_i x_i          <= Storage_Capacity_L      (can't store more than capacity)
               x_i                <= demand_share_i * Market_Demand_L   (per-product demand ceiling)
               x_i                >= 0

Where margin_i (EUR profit per litre for product i) is estimated from the
historical average profit_per_L for that product, and demand_share_i is
each product's historical share of total demand (both estimated from the
same dataset preprocessing.py already builds -- kept consistent with the
rest of the pipeline rather than hand-set).

Solved with scipy.optimize.linprog (HiGHS solver).
"""

import logging
import numpy as np
import pandas as pd
from scipy.optimize import linprog

from .config import PRODUCT_TYPES

logger = logging.getLogger(__name__)


def estimate_margins_and_shares(df: pd.DataFrame) -> tuple:
    """Estimates EUR profit-per-litre and historical demand share for each
    product type from the processed dataframe."""
    margins = df.groupby("Product_Type")["profit_per_L"].mean().reindex(PRODUCT_TYPES).fillna(0)
    counts = df["Product_Type"].value_counts(normalize=True).reindex(PRODUCT_TYPES).fillna(0)
    return margins, counts


def optimize_daily_allocation(
    milk_collection_L: float,
    processing_capacity_L: float,
    storage_capacity_L: float,
    market_demand_L: float,
    margins: pd.Series,
    demand_shares: pd.Series,
) -> dict:
    """
    Solves the single-day LP allocation problem. Returns a dict with the
    optimal litres per product, total predicted profit, and solver status.
    """
    products = list(margins.index)
    n = len(products)

    # linprog MINIMISES by default -> negate margins to maximise profit
    c = -margins.values.astype(float)

    # Aggregate capacity constraints (each row: sum_i x_i <= bound)
    A_ub = [np.ones(n), np.ones(n), np.ones(n)]
    b_ub = [milk_collection_L, processing_capacity_L, storage_capacity_L]

    # Per-product demand ceiling: x_i <= demand_share_i * market_demand_L
    bounds = []
    for p in products:
        cap = max(demand_shares.get(p, 0) * market_demand_L, 0)
        bounds.append((0, cap))

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if not result.success:
        logger.warning("LP did not converge: %s", result.message)
        return {"status": result.message, "success": False}

    allocation = dict(zip(products, np.round(result.x, 1)))
    total_profit = float(-result.fun)

    return {
        "status": "optimal",
        "success": True,
        "allocation_L": allocation,
        "total_predicted_profit_EUR": round(total_profit, 2),
        "milk_utilised_L": round(sum(allocation.values()), 1),
        "milk_unallocated_L": round(milk_collection_L - sum(allocation.values()), 1),
    }


def optimize_from_row(row: pd.Series, margins: pd.Series, demand_shares: pd.Series) -> dict:
    return optimize_daily_allocation(
        milk_collection_L=row["Milk_Collection_L"],
        processing_capacity_L=row["Processing_Capacity_L"],
        storage_capacity_L=row["Storage_Capacity_L"],
        market_demand_L=row["Market_Demand_L"],
        margins=margins,
        demand_shares=demand_shares,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from .preprocessing import load_processed

    df = load_processed()
    margins, shares = estimate_margins_and_shares(df)
    print("Estimated EUR profit-per-litre by product:\n", margins.round(4))
    print("\nHistorical demand share by product:\n", shares.round(3))

    print("\n=== Optimal allocation for the most recent day ===")
    result = optimize_from_row(df.iloc[-1], margins, shares)
    print(result)
