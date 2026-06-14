"""Theme: backend-agnostic colour / style settings for all plot types.

A Theme holds palette names, hatches, markers, and typography in one place.
Built-in presets can be retrieved by name; user-defined themes can be
constructed from a plain dict or JSON and shared across a session.

Usage::

    from omniplot.render.theme import Theme

    t = Theme.default()           # current omniplot defaults
    t = Theme.colorblind()        # colorblind-safe palette
    t = Theme.from_dict({...})    # load from config / JSON
    d = t.to_dict()               # dump for serialisation

Themes are passed *into* backends; they are never imported by backends at
module level so backends stay free of circular dependencies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Theme:
    # ------------------------------------------------------------------ #
    # Colour palettes                                                       #
    # ------------------------------------------------------------------ #

    # Ordered list of matplotlib colourmap names used for successive
    # categorical groups (hue cycles).
    categorical_palettes: list[str] = field(
        default_factory=lambda: [
            "tab20b",
            "tab20c",
            "terrain",
            "nipy_spectral",
            "cubehelix",
            "PiYG",
            "gnuplot",
            "gnuplot2",
        ]
    )

    # Diverging / sequential colourmap for continuous colour scales.
    continuous_palette: str = "coolwarm"

    # Colours used for confidence / prediction interval bands.
    ci_color: str = "skyblue"
    pi_color: str = "lightcyan"
    ci_alpha: float = 0.75

    # Default single-series point / line colour when no category is given.
    point_color: str = "tab:blue"
    line_color: str = "black"

    # Inlier / outlier colours (regression scatter).
    inlier_color: str = "tab:blue"
    outlier_color: str = "tab:red"

    # ------------------------------------------------------------------ #
    # Shape / texture cycling lists                                         #
    # ------------------------------------------------------------------ #

    hatches: list[str] = field(
        default_factory=lambda: ["//", "\\\\", "||", "--", "++", "xx", "oo", "OO", ".."]
    )
    markers: list[str] = field(
        default_factory=lambda: ["o", "^", "s", "D", "v", "p", "*", "h", "X", "P"]
    )
    linestyles: list[str | tuple] = field(
        default_factory=lambda: ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    )

    # ------------------------------------------------------------------ #
    # Typography                                                            #
    # ------------------------------------------------------------------ #

    font_family: str = "sans-serif"
    font_sans_serif: list[str] = field(default_factory=lambda: ["Arial"])
    font_size: float = 14.0

    # ------------------------------------------------------------------ #
    # General aesthetics                                                    #
    # ------------------------------------------------------------------ #

    edgecolor: str = "w"
    point_size: float = 30.0
    point_alpha: float = 1.0

    # ------------------------------------------------------------------ #
    # Constructors / serialisation                                          #
    # ------------------------------------------------------------------ #

    @classmethod
    def default(cls) -> Theme:
        """The standard omniplot look — matches pre-refactor defaults."""
        return cls()

    @classmethod
    def colorblind(cls) -> Theme:
        """Colorblind-safe palette (Okabe–Ito 8-colour cycle)."""
        okabe_ito = [
            "#E69F00",
            "#56B4E9",
            "#009E73",
            "#F0E442",
            "#0072B2",
            "#D55E00",
            "#CC79A7",
            "#000000",
        ]
        return cls(
            categorical_palettes=["tab10"],
            continuous_palette="viridis",
            point_color=okabe_ito[0],
            line_color=okabe_ito[5],
            inlier_color=okabe_ito[2],
            outlier_color=okabe_ito[5],
        )

    @classmethod
    def print(cls) -> Theme:
        """High-contrast, grayscale-friendly theme suitable for print."""
        return cls(
            categorical_palettes=["Greys", "copper", "bone"],
            continuous_palette="RdBu",
            point_color="#222222",
            line_color="#000000",
            edgecolor="k",
        )

    @classmethod
    def dark(cls) -> Theme:
        """Dark-background theme."""
        return cls(
            categorical_palettes=["tab20", "Set1"],
            continuous_palette="plasma",
            ci_color="#1a78c2",
            pi_color="#0d3f6e",
            point_color="#e0e0e0",
            line_color="#ffffff",
            edgecolor="#444444",
        )

    @classmethod
    def from_dict(cls, d: dict) -> Theme:
        """Construct a Theme from a plain dict (e.g. loaded from JSON/YAML)."""
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid}
        return cls(**filtered)

    def to_dict(self) -> dict:
        """Serialise to a plain dict (JSON-safe)."""
        return asdict(self)

    # ------------------------------------------------------------------ #
    # Convenience helpers                                                   #
    # ------------------------------------------------------------------ #

    def categorical_palette(self, index: int = 0) -> str:
        """Return the nth categorical colourmap name (wraps around)."""
        return self.categorical_palettes[index % len(self.categorical_palettes)]
