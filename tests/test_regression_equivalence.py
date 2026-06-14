"""Equivalence (golden) tests for regression_single.

These tests capture the exact numeric outputs of the *current* implementation on
a fixed random seed so that future refactors can be validated against them.

The golden values below were recorded from the following env:
  - sklearn  >= 1.5   (RANSAC)
  - statsmodels >= 0.14  (robust / OLS / elastic_net)
  - numpy >= 1.26

If the underlying library changes its numerics (e.g. a new statsmodels default),
update the golden values deliberately — these tests exist to detect *accidental*
regression, not to freeze upstream numerics forever.
"""

from __future__ import annotations

import importlib

import pytest
import seaborn as sns

from omniplot.scatter import regression_single

BILL_X = "bill_length_mm"
FLIP_Y = "flipper_length_mm"


@pytest.fixture(scope="module")
def penguins_clean():
    return sns.load_dataset("penguins").dropna().reset_index(drop=True)


def _numeric_keys(result: dict) -> dict:
    return {k: float(v) for k, v in result.items() if k not in ("axes", "fitted_model")}


class TestRansacGolden:
    """RANSAC regression golden values."""

    def test_coefficient(self, penguins_clean):
        res = regression_single(
            penguins_clean, x=BILL_X, y=FLIP_Y, method="ransac", random_state=42
        )
        assert _numeric_keys(res)["coefficient"] == pytest.approx(2.53811669, rel=1e-5)

    def test_intercept(self, penguins_clean):
        res = regression_single(
            penguins_clean, x=BILL_X, y=FLIP_Y, method="ransac", random_state=42
        )
        assert _numeric_keys(res)["intercept"] == pytest.approx(94.03779246, rel=1e-5)

    def test_r2(self, penguins_clean):
        res = regression_single(
            penguins_clean, x=BILL_X, y=FLIP_Y, method="ransac", random_state=42
        )
        assert _numeric_keys(res)["r2"] == pytest.approx(0.42653391, rel=1e-5)

    def test_coefficient_pval_is_zero(self, penguins_clean):
        res = regression_single(
            penguins_clean, x=BILL_X, y=FLIP_Y, method="ransac", random_state=42
        )
        assert _numeric_keys(res)["coefficient_pval"] == pytest.approx(0.0, abs=1e-8)

    def test_return_keys(self, penguins_clean):
        res = regression_single(
            penguins_clean, x=BILL_X, y=FLIP_Y, method="ransac", random_state=42
        )
        assert set(res) == {
            "axes",
            "coefficient",
            "intercept",
            "coefficient_pval",
            "r2",
            "fitted_model",
        }


class TestRobustGolden:
    """Robust (RLM) regression golden values."""

    def test_coefficient(self, penguins_clean):
        res = regression_single(
            penguins_clean, x=BILL_X, y=FLIP_Y, method="robust", random_state=42
        )
        assert _numeric_keys(res)["coefficient"] == pytest.approx(1.73914513, rel=1e-5)

    def test_intercept(self, penguins_clean):
        res = regression_single(
            penguins_clean, x=BILL_X, y=FLIP_Y, method="robust", random_state=42
        )
        assert _numeric_keys(res)["intercept"] == pytest.approx(124.66807747, rel=1e-5)

    def test_r2(self, penguins_clean):
        res = regression_single(
            penguins_clean, x=BILL_X, y=FLIP_Y, method="robust", random_state=42
        )
        assert _numeric_keys(res)["r2"] == pytest.approx(0.42653391, rel=1e-5)


class TestOlsGolden:
    """OLS regression golden values."""

    def test_coefficient(self, penguins_clean):
        res = regression_single(penguins_clean, x=BILL_X, y=FLIP_Y, method="ols", random_state=42)
        assert _numeric_keys(res)["coefficient"] == pytest.approx(1.67383257, rel=1e-5)

    def test_intercept(self, penguins_clean):
        res = regression_single(penguins_clean, x=BILL_X, y=FLIP_Y, method="ols", random_state=42)
        assert _numeric_keys(res)["intercept"] == pytest.approx(127.33039758, rel=1e-5)

    def test_r2(self, penguins_clean):
        res = regression_single(penguins_clean, x=BILL_X, y=FLIP_Y, method="ols", random_state=42)
        assert _numeric_keys(res)["r2"] == pytest.approx(0.42653391, rel=1e-5)


class TestElasticNetGolden:
    """Elastic-net (OLS fit_regularized) golden values."""

    def test_coefficient(self, penguins_clean):
        res = regression_single(
            penguins_clean, x=BILL_X, y=FLIP_Y, method="elastic_net", random_state=42
        )
        assert _numeric_keys(res)["coefficient"] == pytest.approx(0.89450028, rel=1e-5)

    def test_intercept(self, penguins_clean):
        res = regression_single(
            penguins_clean, x=BILL_X, y=FLIP_Y, method="elastic_net", random_state=42
        )
        assert _numeric_keys(res)["intercept"] == pytest.approx(162.14360098, rel=1e-5)


class TestLassoGolden:
    """Lasso golden values — requires cvxopt (omniplot[ml] extra)."""

    @pytest.fixture(autouse=True)
    def require_cvxopt(self):
        if importlib.util.find_spec("cvxopt") is None:
            pytest.skip("cvxopt not installed — install omniplot[ml] to run lasso tests")

    def test_runs(self, penguins_clean):
        res = regression_single(penguins_clean, x=BILL_X, y=FLIP_Y, method="lasso", random_state=42)
        assert "coefficient" in res
        assert 0.0 <= res["r2"] <= 1.0
