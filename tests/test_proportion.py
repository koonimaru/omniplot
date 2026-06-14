"""Unit tests for omniplot.proportion."""

import numpy as np
import pandas as pd

from omniplot.proportion import (
    nice_piechart,
    nice_piechart_num,
    stacked_barplot,
    stackedlines,
)


class TestStackedBarplot:
    def test_basic_fraction(self, proportion_df):
        result = stacked_barplot(
            proportion_df,
            x="condition",
            hue="outcome",
            scale="fraction",
        )
        assert "axes" in result

    def test_percentage_scale(self, proportion_df):
        result = stacked_barplot(
            proportion_df,
            x="condition",
            hue="outcome",
            scale="percentage",
        )
        assert "axes" in result

    def test_absolute_scale(self, proportion_df):
        result = stacked_barplot(
            proportion_df,
            x="condition",
            hue="outcome",
            scale="absolute",
        )
        assert "axes" in result

    def test_with_test_pairs(self, proportion_df):
        result = stacked_barplot(
            proportion_df,
            x="condition",
            hue="outcome",
            test_pairs=[["ctrl", "treat"]],
        )
        assert result is not None

    def test_custom_order(self, proportion_df):
        result = stacked_barplot(
            proportion_df,
            x="condition",
            hue="outcome",
            order=["treat", "ctrl"],
        )
        assert "axes" in result


class TestNicePiechart:
    """nice_piechart(df, category) counts occurrences of values in the category column(s)."""

    def test_single_category(self, proportion_df):
        # category is a column name (str) or list of column names
        result = nice_piechart(proportion_df, category="outcome")
        assert result is not None

    def test_multiple_categories(self, proportion_df):
        result = nice_piechart(proportion_df, category=["condition", "outcome"])
        assert result is not None

    def test_with_hatch(self, proportion_df):
        result = nice_piechart(proportion_df, category="outcome", hatch=True)
        assert result is not None


class TestNicePiechartNum:
    """nice_piechart_num(df, variables, category) draws pie charts from numeric values.

    df must be wide-format. Each row becomes one pie chart.
    """

    def test_basic_wide_format(self):
        # Each row = one entity (e.g., a sample); columns = categories
        df = pd.DataFrame(
            {"A": [10.0, 20.0], "B": [30.0, 15.0], "C": [5.0, 25.0]},
            index=["sample1", "sample2"],
        )
        result = nice_piechart_num(df, variables=["A", "B", "C"])
        assert result is not None

    def test_with_category_label(self):
        df = pd.DataFrame(
            {"label": ["s1", "s2"], "A": [10.0, 20.0], "B": [30.0, 15.0]},
        )
        result = nice_piechart_num(df, variables=["A", "B"], category="label")
        assert result is not None


class TestStackedlines:
    """stackedlines(df, x, y) — x is the category column, y is a list of value columns."""

    def test_basic(self, proportion_df):
        # stackedlines expects x=category col, y=list of value cols (numeric)
        # Build a proper wide timeseries-like DataFrame
        rng = np.random.default_rng(42)
        n = 10
        df = pd.DataFrame(
            {
                "timepoint": [f"t{i}" for i in range(n)],
                "group_A": rng.uniform(0, 1, n),
                "group_B": rng.uniform(0, 1, n),
                "group_C": rng.uniform(0, 1, n),
            }
        )
        result = stackedlines(df, x="timepoint", y=["group_A", "group_B", "group_C"])
        assert result is not None

    def test_with_sort_false(self):
        rng = np.random.default_rng(7)
        n = 8
        df = pd.DataFrame(
            {
                "tp": [f"t{i}" for i in range(n)],
                "A": rng.uniform(0, 1, n),
                "B": rng.uniform(0, 1, n),
            }
        )
        result = stackedlines(df, x="tp", y=["A", "B"], sort=False)
        assert result is not None
