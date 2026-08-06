"""
data_validation.py
-------------------
Schema and sanity-check validation for raw input data, run before anything
else touches the pipeline. A decision-support model that silently accepts
malformed input (missing columns, negative volumes, out-of-range
percentages) is a liability -- this module fails loudly and specifically
instead.
"""

import logging
import pandas as pd

from .config import REQUIRED_RAW_COLUMNS, PRODUCT_TYPES

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when the input dataset fails schema or sanity checks."""


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")


def validate_values(df: pd.DataFrame) -> list:
    """Returns a list of warning strings for values that are suspicious but
    not necessarily fatal (out-of-range percentages, negative costs, unknown
    product types, capacity exceeded)."""
    warnings = []

    for col in ["Milk_Collection_L", "Transport_Cost_EUR", "Electricity_Cost_EUR",
                "Processing_Capacity_L", "Storage_Capacity_L", "Market_Demand_L",
                "Market_Price_EUR_per_L", "Yield_kg"]:
        if (df[col] < 0).any():
            warnings.append(f"Negative values found in '{col}'")

    for col in ["Fat_Percent", "Protein_Percent", "Lactose_Percent"]:
        if ((df[col] < 0) | (df[col] > 15)).any():
            warnings.append(f"'{col}' has values outside a plausible 0-15% range")

    unknown_products = set(df["Product_Type"].unique()) - set(PRODUCT_TYPES)
    if unknown_products:
        warnings.append(f"Unrecognised Product_Type values: {unknown_products}")

    over_capacity = df["Milk_Collection_L"] > df["Processing_Capacity_L"]
    if over_capacity.any():
        warnings.append(
            f"{over_capacity.sum()} row(s) have Milk_Collection_L exceeding Processing_Capacity_L"
        )

    if df["Date"].duplicated().any():
        warnings.append("Duplicate dates found in the dataset")

    return warnings


def validate(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """
    Runs schema + value validation. Schema failures always raise.
    Value warnings are logged; if strict=True they also raise.
    """
    validate_schema(df)
    warnings = validate_values(df)
    for w in warnings:
        logger.warning("Data validation warning: %s", w)
    if strict and warnings:
        raise DataValidationError(f"Strict validation failed with {len(warnings)} warning(s): {warnings}")
    return df
