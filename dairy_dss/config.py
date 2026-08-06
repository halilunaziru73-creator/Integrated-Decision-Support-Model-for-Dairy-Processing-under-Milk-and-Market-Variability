"""
config.py
---------
Central configuration: paths, constants, and reproducibility settings.
Keeping these in one place avoids magic numbers/strings scattered across
modules and makes the pipeline easy to retarget (e.g. point at a new CSV,
change random seed for reproducibility audits).
"""

import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "dairy_operations.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

RANDOM_SEED = 42
N_MONTE_CARLO_SIMS = 5000
N_BOOTSTRAP = 2000

SCC_PENALTY_THRESHOLD = 150_000  # cells/mL

PRODUCT_TYPES = ["Cheese", "Butter", "Milk Powder"]

NUMERIC_FEATURES = [
    "milk_quality_index",
    "capacity_utilization",
    "storage_utilization",
    "demand_gap",
    "transport_cost_per_L",
    "electricity_cost_per_L",
    "Market_Price_EUR_per_L",
    "Milk_Collection_L",
]
CATEGORICAL_FEATURES = ["Product_Type"]
TARGET = "Profit_EUR"

REQUIRED_RAW_COLUMNS = [
    "Date", "Milk_Collection_L", "Fat_Percent", "Protein_Percent",
    "Lactose_Percent", "Somatic_Cell_Count", "Transport_Distance_km",
    "Transport_Cost_EUR", "Processing_Capacity_L", "Storage_Capacity_L",
    "Market_Demand_L", "Market_Price_EUR_per_L", "Electricity_Cost_EUR",
    "Product_Type", "Yield_kg", "Profit_EUR",
]


def ensure_output_dirs():
    os.makedirs(FIGURE_DIR, exist_ok=True)
