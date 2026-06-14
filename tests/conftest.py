"""
Shared pytest fixtures for omniplot tests.

All tests use matplotlib's non-interactive 'Agg' backend so they run
headlessly in CI without requiring a display.
"""

import matplotlib

matplotlib.use("Agg")  # must be set before any other matplotlib import

# ---------------------------------------------------------------------------
# Skip legacy script-style test files that require optional chipseq extras.
# These files have no pytest test functions and need pyranges / pyBigWig.
# ---------------------------------------------------------------------------
collect_ignore = [
    "test_chipseq.py",  # requires pyranges, pyBigWig — install omniplot[chipseq]
    "gather_ids.py",  # utility script, not a test
    "tests heatmap.py",  # old ad-hoc script (space in filename)
    "tests_scatter.py",  # old ad-hoc script
]

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import seaborn as sns  # noqa: E402

# ---------------------------------------------------------------------------
# Autouse fixture: close all figures after each test to avoid memory leaks
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Dataset fixtures (built-in seaborn datasets — no external files needed)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def penguins() -> pd.DataFrame:
    """Palmer penguins dataset, NaNs dropped."""
    df = sns.load_dataset("penguins")
    return df.dropna().reset_index(drop=True)


@pytest.fixture(scope="session")
def tips() -> pd.DataFrame:
    """Tips dataset."""
    return sns.load_dataset("tips").reset_index(drop=True)


@pytest.fixture(scope="session")
def iris() -> pd.DataFrame:
    """Iris dataset."""
    return sns.load_dataset("iris").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Synthetic dataset fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def two_group_df() -> pd.DataFrame:
    """Simple two-group long-format DataFrame for catplot / violinplot tests."""
    rng = np.random.default_rng(42)
    n = 50
    return pd.DataFrame(
        {
            "group": ["A"] * n + ["B"] * n,
            "value": np.concatenate([rng.normal(0, 1, n), rng.normal(1, 1, n)]),
        }
    )


@pytest.fixture(scope="session")
def three_group_df() -> pd.DataFrame:
    """Three-group long-format DataFrame."""
    rng = np.random.default_rng(0)
    n = 40
    groups = ["ctrl", "treat1", "treat2"]
    return pd.DataFrame(
        {
            "group": groups * n,
            "value": rng.normal(0, 1, n * 3),
        }
    )


@pytest.fixture(scope="session")
def wide_numeric_df() -> pd.DataFrame:
    """Wide-format DataFrame with 4 numeric variables and a category column."""
    rng = np.random.default_rng(7)
    n = 60
    return pd.DataFrame(
        {
            "cat": (["A"] * 20 + ["B"] * 20 + ["C"] * 20),
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(2, 1, n),
            "x3": rng.normal(-1, 1, n),
            "x4": rng.normal(1, 2, n),
        }
    )


@pytest.fixture(scope="session")
def timeseries_df() -> pd.DataFrame:
    """Timeseries DataFrame for lineplot tests."""
    rng = np.random.default_rng(3)
    t = np.linspace(0, 10, 100)
    return pd.DataFrame(
        {
            "t": t,
            "signal_a": np.sin(t) + rng.normal(0, 0.1, 100),
            "signal_b": np.cos(t) + rng.normal(0, 0.1, 100),
        }
    )


@pytest.fixture(scope="session")
def proportion_df() -> pd.DataFrame:
    """Long-format DataFrame with two categorical columns for proportion plots."""
    rng = np.random.default_rng(11)
    n = 200
    return pd.DataFrame(
        {
            "condition": rng.choice(["ctrl", "treat"], n),
            "outcome": rng.choice(["pos", "neg", "neutral"], n),
        }
    )
