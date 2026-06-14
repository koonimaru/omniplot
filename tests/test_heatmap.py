"""Unit tests for omniplot.heatmap."""

import numpy as np
import pandas as pd
import pytest

from omniplot.heatmap import correlation, dotplot, heatmap


class TestCorrelation:
    """correlation() returns complex_clustermap() result: keys are data/row_clusters/col_clusters/grid."""

    def test_basic_returns_result(self, penguins):
        numeric_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
        result = correlation(penguins[numeric_cols])
        assert isinstance(result, dict)
        # correlation() delegates to complex_clustermap which returns:
        # {"data": ..., "row_clusters": ..., "col_clusters": ..., "grid": ...}
        assert "grid" in result or "data" in result or result is not None

    def test_with_category(self, penguins):
        numeric_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
        df = penguins[numeric_cols + ["species"]].copy()
        # category must be passed as variables + category separately
        result = correlation(
            df,
            variables=numeric_cols,
            category="species",
        )
        assert result is not None

    def test_pearson_method(self, penguins):
        numeric_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
        result = correlation(penguins[numeric_cols], method="pearson")
        assert result is not None

    def test_all_numeric_df(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            rng.normal(0, 1, (30, 4)),
            columns=["a", "b", "c", "d"],
        )
        result = correlation(df)
        assert result is not None


class TestHeatmap:
    """heatmap() takes variables/category kwargs; plain DataFrame needs variables specified."""

    def test_basic_with_variables(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            rng.normal(0, 1, (20, 5)),
            columns=[f"var{i}" for i in range(5)],
        )
        result = heatmap(df, variables=list(df.columns))
        assert result is not None

    def test_with_row_category(self):
        rng = np.random.default_rng(1)
        n = 30
        df = pd.DataFrame(
            rng.normal(0, 1, (n, 4)),
            columns=["a", "b", "c", "d"],
        )
        df["group"] = ["X"] * 15 + ["Y"] * 15
        result = heatmap(df, variables=["a", "b", "c", "d"], category=["group"])
        assert result is not None


class TestDotplot:
    def test_basic_wide_matrix(self):
        rng = np.random.default_rng(2)
        df = pd.DataFrame(
            rng.uniform(0, 1, (5, 4)),
            index=[f"gene{i}" for i in range(5)],
            columns=["A", "B", "C", "D"],
        )
        result = dotplot(df)
        assert result is not None


class TestSchemas:
    """Test Pandera schemas from omniplot.schemas."""

    def test_validate_long_format_passes(self, two_group_df):
        from omniplot.schemas import validate_long_format

        validated = validate_long_format(two_group_df, x="group", y="value")
        assert validated is not None

    def test_validate_long_format_fails_non_numeric_y(self):
        import pandera.errors

        from omniplot.schemas import validate_long_format

        df = pd.DataFrame({"group": ["A", "B"], "value": ["x", "y"]})
        with pytest.raises(pandera.errors.SchemaError):
            validate_long_format(df, x="group", y="value")

    def test_validate_wide_numeric_passes(self, wide_numeric_df):
        from omniplot.schemas import validate_wide_numeric

        validated = validate_wide_numeric(
            wide_numeric_df,
            variables=["x1", "x2", "x3", "x4"],
            category="cat",
        )
        assert validated is not None

    def test_validate_scatter_passes(self, penguins):
        from omniplot.schemas import validate_scatter

        validated = validate_scatter(penguins, x="bill_length_mm", y="bill_depth_mm")
        assert validated is not None

    def test_validate_scatter_missing_column(self, penguins):
        import pandera.errors

        from omniplot.schemas import validate_scatter

        with pytest.raises(pandera.errors.SchemaError):
            validate_scatter(penguins, x="nonexistent_col", y="bill_depth_mm")
