"""omniplot.render — backend-agnostic rendering layer.

Public surface:

    from omniplot.render import Theme, PlotSpec, MatplotlibBackend, RenderResult

Internal sub-modules (not part of the stable public API yet):
    omniplot.render.theme    — Theme dataclass + presets
    omniplot.render.spec     — PlotSpec, layer dataclasses, RegressionResult
    omniplot.render.backend  — Backend protocol, RenderResult
    omniplot.render.backends.matplotlib_backend — MatplotlibBackend
"""

from omniplot.render.backend import Backend, RenderResult
from omniplot.render.backends.matplotlib_backend import MatplotlibBackend
from omniplot.render.spec import (
    AxisLabelLayer,
    BandLayer,
    Layer,
    LineLayer,
    PlotSpec,
    PointsLayer,
    RegressionResult,
    build_regression_spec,
)
from omniplot.render.theme import Theme

__all__ = [
    # Theme
    "Theme",
    # Spec / IR
    "PlotSpec",
    "Layer",
    "PointsLayer",
    "LineLayer",
    "BandLayer",
    "AxisLabelLayer",
    "RegressionResult",
    "build_regression_spec",
    # Backend
    "Backend",
    "RenderResult",
    "MatplotlibBackend",
]
