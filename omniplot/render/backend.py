"""Backend protocol — the interface every rendering backend must implement.

A backend is responsible for translating a PlotSpec + Theme into actual
drawing calls (matplotlib figure, R script, Plotly figure, …).

Implementing a new backend
--------------------------
1.  Subclass ``Backend`` (or implement the Protocol structurally).
2.  Override ``render(spec, theme, ax)`` to produce the figure.
3.  Return a ``RenderResult`` with the figure handle, axes handle, and any
    extra backend-specific objects the caller might want.

Currently shipped backends
--------------------------
- ``MatplotlibBackend`` — wraps matplotlib / seaborn primitives.
  See ``omniplot.render.backends.matplotlib_backend``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from omniplot.render.spec import PlotSpec
from omniplot.render.theme import Theme


@dataclass
class RenderResult:
    """Container returned by every backend after rendering.

    Attributes
    ----------
    figure:
        The top-level figure object in whatever type the backend produces.
        For ``MatplotlibBackend`` this is a ``matplotlib.figure.Figure``;
        for ``RBackend`` it would be the path to the saved image.
    axes:
        The primary axes / panel object.  May be ``None`` for backends that
        don't expose an axes concept (e.g. R).
    extra:
        Backend-specific extras (e.g. seaborn FacetGrid, igraph layout, …).
    """

    figure: Any
    axes: Any
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Backend(Protocol):
    """Structural protocol for plot backends.

    Any class that implements ``render(spec, theme, ax)`` conforms to this
    protocol automatically — no explicit subclassing required.
    """

    def render(
        self,
        spec: PlotSpec,
        theme: Theme,
        ax: Any | None = None,
    ) -> RenderResult:
        """Render *spec* using *theme* and return a ``RenderResult``.

        Parameters
        ----------
        spec:
            The backend-agnostic plot description.
        theme:
            Visual style settings.
        ax:
            Optional pre-existing axes / canvas to draw onto.
            When ``None`` the backend creates a new figure.

        Returns
        -------
        RenderResult
        """
        ...

    def save(self, result: RenderResult, path: str, suffix: str = "") -> None:
        """Persist *result* to *path*.

        The concrete meaning of "save" is backend-specific.  Matplotlib saves
        a PDF/PNG/SVG; the R backend may move/rename an already-written file.
        *suffix* is appended to the stem before the extension (matching the
        existing ``_save`` behaviour).
        """
        ...
