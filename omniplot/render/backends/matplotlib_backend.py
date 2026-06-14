"""MatplotlibBackend — renders a PlotSpec using matplotlib + seaborn.

This backend wraps the existing low-level rendering helpers from
``omniplot.utils`` (``_draw_ci_pi``, ``_create_color_markerlut``, …) so that
migrated functions produce identical output to the legacy code while the
internal plumbing is progressively restructured.

Layer dispatch
--------------
Each ``Layer`` subtype is handled by a private ``_draw_*`` method.  New layer
types added to ``spec.py`` during Phase 4 migration should be paired with a
corresponding ``_draw_*`` method here.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

from omniplot.render.backend import RenderResult
from omniplot.render.spec import (
    AxisLabelLayer,
    BandLayer,
    Layer,
    LineLayer,
    PlotSpec,
    PointsLayer,
)
from omniplot.render.theme import Theme


class MatplotlibBackend:
    """Renders a ``PlotSpec`` to a matplotlib ``Figure``/``Axes`` pair.

    Conforms to the ``Backend`` protocol — no inheritance needed.
    """

    # ------------------------------------------------------------------ #
    # Public interface (Backend protocol)                                  #
    # ------------------------------------------------------------------ #

    def render(
        self,
        spec: PlotSpec,
        theme: Theme,
        ax: plt.Axes | None = None,
    ) -> RenderResult:
        """Render *spec* onto *ax* (or a freshly created axes) and return the result."""
        plt.rcParams.update(
            {
                "font.size": theme.font_size,
                "font.family": theme.font_family,
                "font.sans-serif": theme.font_sans_serif,
            }
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=spec.figsize)
            plt.subplots_adjust(left=0.15)
        else:
            fig = ax.get_figure()

        if spec.title:
            ax.set_title(spec.title)
        if spec.xlabel:
            ax.set_xlabel(spec.xlabel)
        if spec.ylabel:
            ax.set_ylabel(spec.ylabel)

        for layer in spec.layers:
            self._draw_layer(layer, ax, theme)

        if spec.show_legend:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels)

        return RenderResult(figure=fig, axes=ax)

    def save(self, result: RenderResult, path: str, suffix: str = "") -> None:
        """Save *result.figure* to *path* with an optional *suffix* appended to the stem."""
        if not path:
            return
        fig: Figure = result.figure
        if path.endswith((".pdf", ".png", ".svg")):
            stem, ext = os.path.splitext(path)
            out = f"{stem}_{suffix}{ext}" if suffix else f"{stem}{ext}"
        else:
            out = f"{path}_{suffix}.pdf" if suffix else f"{path}.pdf"
        fig.savefig(out)

    # ------------------------------------------------------------------ #
    # Layer dispatch                                                        #
    # ------------------------------------------------------------------ #

    def _draw_layer(self, layer: Layer, ax: plt.Axes, theme: Theme) -> None:
        dispatch: dict[type, Callable[..., None]] = {
            PointsLayer: self._draw_points,
            LineLayer: self._draw_line,
            BandLayer: self._draw_band,
            AxisLabelLayer: self._draw_axis_label,
        }
        handler = dispatch.get(type(layer))
        if handler is None:
            raise NotImplementedError(
                f"MatplotlibBackend does not yet handle layer type {type(layer).__name__!r}. "
                "Add a _draw_* method and register it in _draw_layer."
            )
        handler(layer, ax, theme)

    def _draw_points(self, layer: PointsLayer, ax: plt.Axes, theme: Theme) -> None:
        if layer.groups is not None:
            # Delegate to seaborn for categorical hue handling.
            import pandas as pd

            df_tmp = pd.DataFrame({"x": layer.x, "y": layer.y, "_hue": layer.groups})
            sns.scatterplot(
                data=df_tmp,
                x="x",
                y="y",
                hue="_hue",
                ax=ax,
                palette=theme.categorical_palette(0),
                alpha=layer.alpha,
                s=layer.size,
            )
        else:
            color = layer.color or theme.point_color
            ax.scatter(
                layer.x,
                layer.y,
                color=color,
                label=layer.label,
                alpha=layer.alpha,
                s=layer.size,
                edgecolors=theme.edgecolor,
                linewidths=0.5,
            )

    def _draw_line(self, layer: LineLayer, ax: plt.Axes, theme: Theme) -> None:
        color = layer.color or theme.line_color
        ax.plot(
            layer.x,
            layer.y,
            color=color,
            label=layer.label,
            linewidth=layer.linewidth,
            linestyle=layer.linestyle,
        )

    def _draw_band(self, layer: BandLayer, ax: plt.Axes, theme: Theme) -> None:
        ax.fill_between(
            layer.x,
            layer.y_lower,
            layer.y_upper,
            color=layer.color,
            alpha=layer.alpha,
            label=layer.label,
        )

    def _draw_axis_label(self, layer: AxisLabelLayer, ax: plt.Axes, theme: Theme) -> None:
        if layer.axis == "y":
            ax.text(0, 1, layer.text, transform=ax.transAxes, ha="right", va="bottom")
        else:
            ax.text(1, 0, layer.text, transform=ax.transAxes, ha="left", va="top")


# ------------------------------------------------------------------ #
# Module-level default instance (convenience)                          #
# ------------------------------------------------------------------ #

#: A ready-to-use default instance; callers can import and use directly:
#:   ``from omniplot.render.backends.matplotlib_backend import mpl``
mpl = MatplotlibBackend()
