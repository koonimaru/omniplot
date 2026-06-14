"""Unit tests for omniplot.scatter."""

from omniplot.scatter import decomplot, manifoldplot, regression_single, scatterplot


class TestScatterplot:
    def test_basic(self, penguins):
        result = scatterplot(
            penguins,
            x="bill_length_mm",
            y="bill_depth_mm",
        )
        assert result is not None

    def test_with_category(self, penguins):
        result = scatterplot(
            penguins,
            x="bill_length_mm",
            y="bill_depth_mm",
            category=["species"],
        )
        assert result is not None

    def test_with_multiple_categories(self, penguins):
        result = scatterplot(
            penguins,
            x="bill_length_mm",
            y="bill_depth_mm",
            category=["species", "sex"],
        )
        assert result is not None


class TestDecomplot:
    """Test dimensionality reduction plots."""

    def test_pca(self, penguins):
        variables = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
        result = decomplot(penguins, variables=variables, method="pca")
        assert result is not None

    def test_nmf(self, penguins):
        variables = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
        # NMF requires non-negative values — shift data
        df = penguins.copy()
        for v in variables:
            df[v] = df[v] - df[v].min() + 0.01
        # Pass category=[] (empty list) not "" to avoid seaborn hue error
        result = decomplot(df, variables=variables, method="nmf")
        assert result is not None

    def test_with_category(self, penguins):
        variables = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
        result = decomplot(penguins, variables=variables, category="species", method="pca")
        assert result is not None


class TestManifoldplot:
    def test_tsne(self, penguins):
        variables = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
        # n_iter renamed to max_iter in sklearn >= 1.5 — fixed in utils.py
        result = manifoldplot(
            penguins,
            variables=variables,
            category=["species"],
            method="tsne",
        )
        assert result is not None

    def test_mds(self, penguins):
        variables = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
        result = manifoldplot(
            penguins,
            variables=variables,
            method="mds",
        )
        assert result is not None


class TestRegressionSingle:
    """regression_single returns {"axes", "coefficient", "intercept", "coefficient_pval", "r2", "fitted_model"}."""

    def test_ransac(self, penguins):
        result = regression_single(
            penguins,
            x="bill_length_mm",
            y="flipper_length_mm",
            method="ransac",
        )
        assert "axes" in result
        assert "r2" in result

    def test_robust(self, penguins):
        # Fixed: fitted_model.params.iloc[1] instead of params[1]
        result = regression_single(
            penguins,
            x="bill_length_mm",
            y="flipper_length_mm",
            method="robust",
        )
        assert "axes" in result
        assert "r2" in result

    def test_with_category(self, penguins):
        result = regression_single(
            penguins,
            x="bill_length_mm",
            y="flipper_length_mm",
            method="ransac",
            category="species",
        )
        assert "axes" in result

    def test_r2_range(self, penguins):
        result = regression_single(
            penguins,
            x="bill_length_mm",
            y="flipper_length_mm",
        )
        assert 0 <= result["r2"] <= 1
