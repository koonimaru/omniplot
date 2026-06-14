"""Unit tests for omniplot.utils."""

import numpy as np
import pandas as pd
import pytest

from omniplot.utils import (
    _calc_curveture,
    _calc_r2,
    _ci_pi,
    _create_color_markerlut,
    _dendrogram_threshold,
    _get_cluster_classes,
    _separate_data,
    colormap_list,
    hatch_list,
    marker_list,
)


class TestConstants:
    def test_colormap_list_nonempty(self):
        assert len(colormap_list) > 0
        assert all(isinstance(c, str) for c in colormap_list)

    def test_hatch_list_nonempty(self):
        assert len(hatch_list) > 0

    def test_marker_list_nonempty(self):
        assert len(marker_list) > 0


class TestCalcR2:
    def test_perfect_correlation(self):
        x = np.array([1.0, 2, 3, 4, 5])
        r2 = _calc_r2(x, x)
        assert abs(r2 - 1.0) < 1e-10

    def test_zero_correlation(self):
        rng = np.random.default_rng(0)
        x = np.arange(100, dtype=float)
        y = rng.normal(0, 1, 100)
        # R² should be small (not necessarily exactly 0)
        r2 = _calc_r2(x, y)
        assert r2 < 0.5

    def test_negative_correlation(self):
        x = np.array([1.0, 2, 3, 4, 5])
        r2 = _calc_r2(x, -x)
        assert abs(r2 - 1.0) < 1e-10


class TestCiPi:
    def test_output_shapes(self):
        rng = np.random.default_rng(1)
        n = 50
        X = np.arange(n, dtype=float)
        Y = 2.0 * X + rng.normal(0, 1, n)
        plotline_X = X.reshape(-1, 1)
        y_model = 2.0 * X
        ci, pi, std_err = _ci_pi(X, Y, plotline_X, y_model)
        assert ci.shape == (n,)
        assert pi.shape == (n,)
        assert pi.mean() > ci.mean()  # PI must be wider than CI

    def test_pi_wider_than_ci(self):
        rng = np.random.default_rng(2)
        X = np.linspace(0, 10, 100)
        Y = 3.0 * X + rng.normal(0, 2, 100)
        plotline_X = X.reshape(-1, 1)
        y_model = 3.0 * X
        ci, pi, _ = _ci_pi(X, Y, plotline_X, y_model)
        assert np.all(pi >= ci)


class TestSeparateData:
    def test_variables_specified(self, wide_numeric_df):
        variables = ["x1", "x2", "x3", "x4"]
        x, cat = _separate_data(wide_numeric_df, variables=variables, category="cat")
        assert x.shape == (len(wide_numeric_df), 4)
        assert cat == ["cat"]

    def test_no_variables_raises_on_non_numeric(self):
        df = pd.DataFrame({"a": ["x", "y"], "b": [1.0, 2.0]})
        with pytest.raises(TypeError):
            _separate_data(df)


class TestCreateColorMarkerlut:
    def test_string_palette(self, penguins):
        lut, mlut = _create_color_markerlut(penguins, "species", "tab10")
        species = penguins["species"].unique()
        assert set(lut.keys()) == set(species)

    def test_dict_palette(self, penguins):
        palette = {"Adelie": "red", "Chinstrap": "blue", "Gentoo": "green"}
        lut, _ = _create_color_markerlut(penguins, "species", palette)
        assert lut == palette

    def test_invalid_palette_type(self, penguins):
        with pytest.raises(TypeError):
            _create_color_markerlut(penguins, "species", 42)


class TestDendrogramThreshold:
    def test_returns_float(self, penguins):
        import scipy.cluster.hierarchy as sch
        from scipy.spatial.distance import pdist, squareform
        from scipy.stats import zscore

        features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
        X = zscore(penguins[features].values, axis=0)
        D = squareform(pdist(X, metric="euclidean"))
        Y = sch.linkage(D, method="ward")
        Z = sch.dendrogram(Y, labels=list(penguins.index), no_plot=True)
        t = _dendrogram_threshold(Z, 3)
        assert isinstance(t, float)
        assert t >= 0


class TestGetClusterClasses:
    def test_returns_dict(self, penguins):
        import scipy.cluster.hierarchy as sch
        from scipy.spatial.distance import pdist, squareform
        from scipy.stats import zscore

        features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
        X = zscore(penguins[features].values, axis=0)
        D = squareform(pdist(X, metric="euclidean"))
        Y = sch.linkage(D, method="ward")
        Z = sch.dendrogram(Y, labels=list(range(len(penguins))), no_plot=True)
        _dendrogram_threshold(Z, 3)
        clusters = _get_cluster_classes(Z)
        assert isinstance(clusters, dict)
        assert len(clusters) >= 1


class TestCalcCurveture:
    def test_output_length(self):
        x = np.linspace(0, 1, 10)
        y = x**2
        perp = _calc_curveture(x, y)
        assert len(perp) == len(x)
        assert perp[0] == 0  # first element is always 0
