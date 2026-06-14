"""Unit tests for omniplot.plot (catplot, violinplot2, lineplot, radialtree)."""

import numpy as np
import pytest

import omniplot as op
from omniplot.plot import catplot, lineplot, radialtree, violinplot2


class TestCatplot:
    def test_returns_dict_with_axes(self, penguins):
        result = catplot(penguins, x="species", y="bill_length_mm")
        assert "axes" in result
        assert "p values" in result

    def test_with_pairs(self, penguins):
        pairs = [["Adelie", "Chinstrap"], ["Adelie", "Gentoo"]]
        result = catplot(
            penguins,
            x="species",
            y="bill_length_mm",
            pairs=pairs,
            test="ttest_ind",
        )
        assert "Adelie_Chinstrap" in result["p values"]
        assert "Adelie_Gentoo" in result["p values"]

    def test_box_kind(self, penguins):
        result = catplot(penguins, x="species", y="bill_length_mm", kind="box")
        assert result["axes"] is not None

    def test_invalid_test_raises(self, penguins):
        with pytest.raises(ValueError, match="Available tests"):
            catplot(penguins, x="species", y="bill_length_mm", test="bad_test")

    def test_significance_symbol(self, two_group_df):
        pairs = [["A", "B"]]
        result = catplot(
            two_group_df,
            x="group",
            y="value",
            pairs=pairs,
            significance="symbol",
        )
        assert "axes" in result

    def test_mannwhitneyu(self, two_group_df):
        pairs = [["A", "B"]]
        result = catplot(
            two_group_df,
            x="group",
            y="value",
            pairs=pairs,
            test="mannwhitneyu",
        )
        assert "A_B" in result["p values"]


class TestViolinplot2:
    def test_dataframe_input(self, two_group_df):
        result = violinplot2(two_group_df, x="group", y="value")
        assert "axes" in result

    def test_with_pairs(self, two_group_df):
        result = violinplot2(
            two_group_df,
            x="group",
            y="value",
            pairs=[["A", "B"]],
        )
        assert "p values" in result

    def test_numpy_input(self):
        rng = np.random.default_rng(5)
        data = [rng.normal(0, 1, 30), rng.normal(1, 1, 30)]
        result = violinplot2(data, xlabels=["A", "B"])
        assert "axes" in result

    def test_horizontal_orientation(self, two_group_df):
        result = violinplot2(two_group_df, x="group", y="value", orientation="horizontal")
        assert result["axes"] is not None


class TestLineplot:
    def test_single_variable(self, timeseries_df):
        result = lineplot(timeseries_df, x="t", y="signal_a")
        assert "axes" in result

    def test_multiple_variables(self, timeseries_df):
        result = lineplot(timeseries_df, x="t", y=["signal_a", "signal_b"])
        assert "axes" in result

    def test_no_y_raises(self, timeseries_df):
        with pytest.raises(ValueError):
            lineplot(timeseries_df, x="t")

    def test_split_subplots(self, timeseries_df):
        result = lineplot(
            timeseries_df,
            x="t",
            y=["signal_a", "signal_b"],
            split=True,
        )
        assert "axes" in result

    def test_moving_average_estimate(self, timeseries_df):
        result = lineplot(timeseries_df, x="t", y="signal_a", estimate="moving_average")
        assert result["axes"] is not None


class TestRadialtree:
    def test_returns_dict(self, penguins):
        result = radialtree(
            penguins,
            n_clusters=3,
            variables=["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"],
            category=["species"],
        )
        assert "axes" in result
        assert "clusters" in result

    def test_clusters_is_dict(self, penguins):
        result = radialtree(
            penguins,
            n_clusters=3,
            variables=["bill_length_mm", "bill_depth_mm"],
            category=["species"],
        )
        assert isinstance(result["clusters"], dict)

    def test_multiple_categories(self, penguins):
        result = radialtree(
            penguins,
            n_clusters=3,
            variables=["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"],
            category=["species", "sex"],
        )
        assert "axes" in result


class TestPublicAPI:
    """Smoke-test the top-level omniplot namespace."""

    def test_catplot_importable(self):
        assert callable(op.catplot)

    def test_radialtree_importable(self):
        assert callable(op.radialtree)

    def test_lineplot_importable(self):
        assert callable(op.lineplot)
