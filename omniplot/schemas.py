"""
Pandera DataFrame schemas for omniplot input validation.

Column names in plotting functions are user-supplied strings, so schemas here
are defined as *validators* (functions) rather than fixed DataFrameModel
subclasses.  Each validator raises pandera.errors.SchemaError on invalid input.
"""

from __future__ import annotations

import pandas as pd
import pandera as pa


def validate_long_format(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    y_dtype: type = float,
    nullable: bool = True,
) -> pd.DataFrame:
    """Validate a long-format DataFrame used by catplot / violinplot.

    The *x* column may be any type (categorical labels).
    The *y* column must be numeric (float or int).
    """
    schema = pa.DataFrameSchema(
        {
            x: pa.Column(nullable=nullable),
            y: pa.Column(float, nullable=nullable),
        },
        strict=False,  # extra columns are allowed
    )
    return schema.validate(df)


def validate_wide_numeric(
    df: pd.DataFrame,
    variables: list[str],
    category: str | list[str] | None = None,
    *,
    nullable: bool = True,
) -> pd.DataFrame:
    """Validate a wide-format DataFrame where *variables* columns must be numeric.

    Used by radialtree, correlation, decomplot, manifoldplot, etc.
    """
    cols: dict[str, pa.Column] = {v: pa.Column(float, nullable=nullable) for v in variables}
    if category:
        cats = [category] if isinstance(category, str) else category
        for c in cats:
            cols[c] = pa.Column(nullable=nullable)

    schema = pa.DataFrameSchema(cols, strict=False)
    return schema.validate(df)


def validate_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    nullable: bool = True,
) -> pd.DataFrame:
    """Validate a DataFrame for scatter / regression plots.

    Both *x* and *y* columns must be numeric.
    """
    schema = pa.DataFrameSchema(
        {
            x: pa.Column(float, nullable=nullable),
            y: pa.Column(float, nullable=nullable),
        },
        strict=False,
    )
    return schema.validate(df)


def validate_stacked_bar(
    df: pd.DataFrame,
    x: str,
    hue: str,
    *,
    nullable: bool = True,
) -> pd.DataFrame:
    """Validate a DataFrame for stacked_barplot / nice_piechart.

    Both *x* and *hue* must be present; their values are treated as categories.
    """
    schema = pa.DataFrameSchema(
        {
            x: pa.Column(nullable=nullable),
            hue: pa.Column(nullable=nullable),
        },
        strict=False,
    )
    return schema.validate(df)


# ---------------------------------------------------------------------------
# Concrete DataFrameModel subclasses for fully-typed internal pipelines
# ---------------------------------------------------------------------------


class HeatmapDataFrame(pa.DataFrameModel):
    """Schema for a numeric matrix used in heatmap / clustermap functions.

    All columns are expected to be floating-point.  Row index is unconstrained.
    Use ``validate_wide_numeric`` instead when column names are user-supplied.
    """

    class Config:
        strict = False
        dtype_backend = "numpy_nullable"


class CorrelationResultDataFrame(pa.DataFrameModel):
    """Schema for the correlation-coefficient matrix returned by ``correlation``."""

    class Config:
        strict = False
