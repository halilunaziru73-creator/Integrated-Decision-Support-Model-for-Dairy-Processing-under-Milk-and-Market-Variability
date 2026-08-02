"""
preprocessing.py
-----------------
Loads raw daily dairy operations data, validates it, and engineers the
features used by the predictive model and decision engine.

Engineered features:
    - milk_quality_index      : weighted composite of fat/protein/lactose,
                                 penalised by somatic cell count (SCC).
                                 Higher = better raw milk quality.
    - capacity_utilization    : Milk_Collection_L / Processing_Capacity_L
    - storage_utilization     : Milk_Collection_L / Storage_Capacity_L
    - demand_gap              : Market_Demand_L - Milk_Collection_L
                                 (positive => demand exceeds supply)
    - transport_cost_per_L    : Transport_Cost_EUR / Milk_Collection_L
    - electricity_cost_per_L  : Electricity_Cost_EUR / Milk_Collection_L
    - revenue_potential       : Milk_Collection_L * Market_Price_EUR_per_L
"""

import logging
import pandas as pd
import numpy as np

from config import DATA_PATH, SCC_PENALTY_THRESHOLD
from data_validation import validate

logger = logging.getLogger(__name__)


def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    logger.info("Loading raw data from %s", path)
    df = pd.read_csv(path, parse_dates=["Date"])
    validate(df)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Composite raw-milk quality index (illustrative weighting based on
    # standard dairy-quality payment schemes: fat and protein content drive
    # processing yield; SCC above the regulatory-adjacent threshold reduces
    # usable quality).
    scc_penalty = np.clip((df["Somatic_Cell_Count"] - SCC_PENALTY_THRESHOLD) / 10_000, 0, None)
    df["milk_quality_index"] = (
        df["Fat_Percent"] * 10
        + df["Protein_Percent"] * 8
        + df["Lactose_Percent"] * 2
        - scc_penalty
    )

    df["capacity_utilization"] = df["Milk_Collection_L"] / df["Processing_Capacity_L"]
    df["storage_utilization"] = df["Milk_Collection_L"] / df["Storage_Capacity_L"]
    df["demand_gap"] = df["Market_Demand_L"] - df["Milk_Collection_L"]
    df["transport_cost_per_L"] = df["Transport_Cost_EUR"] / df["Milk_Collection_L"]
    df["electricity_cost_per_L"] = df["Electricity_Cost_EUR"] / df["Milk_Collection_L"]
    df["revenue_potential"] = df["Milk_Collection_L"] * df["Market_Price_EUR_per_L"]
    df["profit_per_L"] = df["Profit_EUR"] / df["Milk_Collection_L"]
    df["yield_ratio"] = df["Yield_kg"] / df["Milk_Collection_L"]

    logger.info("Feature engineering complete: %d rows, %d columns", *df.shape)
    return df


def load_processed(path: str = DATA_PATH) -> pd.DataFrame:
    return engineer_features(load_raw(path))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = load_processed()
    print(data.round(3).to_string(index=False))
