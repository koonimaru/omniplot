"""Plot specification (intermediate representation).

A PlotSpec describes *what* to draw — layers, encodings, labels — without
reference to any rendering library.  Backends consume PlotSpec to produce
actual figures.

Only the layer types needed for the regression pilot are defined here.
New layer types will be added in the same module as each plot family is
migrated (Phase 4 of the migration plan).

Layer hierarchy
---------------
Layer (Union alias)
  PointsLayer   — scatter points, optionally grouped by a category column
  LineLayer     — a single line (e.g. regression line)
  BandLayer     — filled region between two y-arrays (CI / PI ribbons)
  AxisLabelLayer — unit annotation next to an axis label
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ------------------------------------------------------------------ #
# Individual layers                                                    #
# ------------------------------------------------------------------ #


@dataclass
class PointsLayer:
    """Scatter points drawn from pre-extracted arrays.

    When *groups* is provided (a string array of the same length as *x*/*y*),
    each unique group value is drawn in a distinct colour from the theme's
    categorical palette.
    """

    x: np.ndarray
    y: np.ndarray
    color: str | None = None
    label: str | None = None
    groups: np.ndarray | None = None  # categorical hue — overrides color
    alpha: float = 1.0
    size: float = 30.0


@dataclass
class LineLayer:
    """A single continuous line (e.g. fitted regression line)."""

    x: np.ndarray
    y: np.ndarray
    color: str | None = None
    label: str | None = None
    linewidth: float = 1.5
    linestyle: str = "-"


@dataclass
class BandLayer:
    """Filled band between *y_lower* and *y_upper* (confidence / prediction intervals)."""

    x: np.ndarray
    y_lower: np.ndarray
    y_upper: np.ndarray
    color: str = "skyblue"
    alpha: float = 0.75
    label: str | None = None


@dataclass
class AxisLabelLayer:
    """Unit annotation placed adjacent to an axis label.

    *axis* is "x" or "y"; *text* is the unit string (e.g. "mm" or "g").
    The backend decides exact placement to avoid overlapping tick labels.
    """

    axis: str  # "x" | "y"
    text: str


# Extend this union as more layer types are added in later phases.
Layer = PointsLayer | LineLayer | BandLayer | AxisLabelLayer


# ------------------------------------------------------------------ #
# Plot specification                                                   #
# ------------------------------------------------------------------ #


@dataclass
class PlotSpec:
    """Complete, backend-agnostic specification of a single plot panel.

    Attributes
    ----------
    layers:
        Ordered list of layers; backends draw them in order.
    title:
        Panel title string (empty → no title).
    xlabel / ylabel:
        Axis label strings.
    figsize:
        (width, height) in inches — a hint to the backend; backends may
        ignore this when *ax* is supplied externally.
    show_legend:
        Whether to draw a legend.
    """

    layers: list[Layer] = field(default_factory=list)
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    figsize: tuple[float, float] = (5.0, 5.0)
    show_legend: bool = True

    def add(self, layer: Layer) -> PlotSpec:
        """Append *layer* and return self for fluent chaining."""
        self.layers.append(layer)
        return self


# ------------------------------------------------------------------ #
# Convenience builder for regression plots                             #
# ------------------------------------------------------------------ #


@dataclass
class RegressionResult:
    """Pure-data result of a regression fit — no matplotlib objects.

    Returned by the stats helpers and consumed by spec builders / backends.
    The *fitted_model* is the raw statsmodels / sklearn object so callers can
    call ``fitted_model.predict`` directly.
    """

    method: str
    coefficient: float
    intercept: float
    coefficient_pval: float
    r2: float
    mse: float
    x_line: np.ndarray
    y_line: np.ndarray
    ci: np.ndarray  # confidence interval half-width at each x_line point
    pi: np.ndarray  # prediction interval half-width
    std_error: float
    inlier_mask: np.ndarray | None = None  # RANSAC only
    outlier_mask: np.ndarray | None = None  # RANSAC only
    fitted_model: object = None


def build_regression_spec(
    x_data: np.ndarray,
    y_data: np.ndarray,
    result: RegressionResult,
    *,
    xlabel: str = "",
    ylabel: str = "",
    figsize: tuple[float, float] = (5.0, 5.0),
    xunit: str = "",
    yunit: str = "",
    groups: np.ndarray | None = None,
    point_color: str = "tab:blue",
    line_color: str = "tab:orange",
    ci_color: str = "skyblue",
    pi_color: str = "lightcyan",
    ci_alpha: float = 0.75,
) -> PlotSpec:
    """Assemble a PlotSpec for a single-panel regression plot.

    Backend-agnostic: takes arrays and a RegressionResult, returns a PlotSpec
    that any backend can render.
    """
    r = result
    method_labels = {
        "ransac": "RANSAC regression",
        "robust": "Robust linear regression",
        "ols": "OLS",
        "lasso": "Lasso",
        "elastic_net": "Elastic net",
    }
    method_name = method_labels.get(r.method, r.method)
    title = (
        f"{method_name}, r²={r.r2:.2f}, MSE={r.mse:.2f}\n"
        f"y = {r.intercept:.2f} + {r.coefficient:.2f}x, "
        f"p-value: {r.coefficient_pval:.2E}"
    )

    spec = PlotSpec(title=title, xlabel=xlabel, ylabel=ylabel, figsize=figsize)

    # CI / PI bands — drawn first so points appear on top
    spec.add(
        BandLayer(
            x=r.x_line,
            y_lower=r.y_line - r.pi,
            y_upper=r.y_line + r.pi,
            color=pi_color,
            alpha=ci_alpha * 0.5,
            label="95% prediction interval",
        )
    )
    spec.add(
        BandLayer(
            x=r.x_line,
            y_lower=r.y_line - r.ci,
            y_upper=r.y_line + r.ci,
            color=ci_color,
            alpha=ci_alpha,
            label="95% confidence interval",
        )
    )

    # Points — groups (categorical hue) takes priority; fall back to
    # inlier/outlier split for RANSAC, then plain single-colour scatter.
    if groups is not None:
        spec.add(PointsLayer(x=x_data, y=y_data, groups=groups))
    elif r.inlier_mask is not None and r.outlier_mask is not None:
        spec.add(
            PointsLayer(
                x=x_data[r.inlier_mask], y=y_data[r.inlier_mask], color=point_color, label="Inliers"
            )
        )
        spec.add(
            PointsLayer(
                x=x_data[r.outlier_mask],
                y=y_data[r.outlier_mask],
                color="tab:red",
                label="Outliers",
            )
        )
    else:
        spec.add(PointsLayer(x=x_data, y=y_data, color=point_color))

    # Regression line
    spec.add(LineLayer(x=r.x_line, y=r.y_line, color=line_color))

    # Unit annotations
    if yunit:
        spec.add(AxisLabelLayer(axis="y", text=f"({yunit})"))
    if xunit:
        spec.add(AxisLabelLayer(axis="x", text=f"({xunit})"))

    return spec
