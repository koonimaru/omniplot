"""
omniplot — R-like statistical plotting for Python researchers.

Public API is re-exported here so users can do::

    import omniplot as op
    op.catplot(df, x="species", y="bill_length_mm")
"""

from omniplot._version import __version__

# Heatmap / cluster
from omniplot.heatmap import (
    complex_clustermap,
    correlation,
    dotplot,
    heatmap,
    triangle_heatmap,
)

# Core plotting
from omniplot.plot import (
    catplot,
    lineplot,
    radialtree,
    violinplot,
    violinplot2,
)

# Proportion / composition
from omniplot.proportion import (
    nested_piechart,
    nice_piechart,
    nice_piechart_num,
    stacked_barplot,
    stacked_barplot_num,
    stackedlines,
)

# Regression
from omniplot.regressionplot import regression_single as regression_single_robust  # noqa: F401

# Scatter / dimensionality reduction
from omniplot.scatter import (
    clusterplot,
    decomplot,
    manifoldplot,
    pie_scatter,
    regression_single,
    scatterplot,
    volcanoplot,
)

__all__ = [
    "__version__",
    # plot
    "catplot",
    "lineplot",
    "radialtree",
    "violinplot",
    "violinplot2",
    # heatmap
    "complex_clustermap",
    "correlation",
    "dotplot",
    "heatmap",
    "triangle_heatmap",
    # scatter
    "clusterplot",
    "decomplot",
    "manifoldplot",
    "pie_scatter",
    "regression_single",
    "scatterplot",
    "volcanoplot",
    # proportion
    "nested_piechart",
    "nice_piechart",
    "nice_piechart_num",
    "stacked_barplot",
    "stacked_barplot_num",
    "stackedlines",
]

# Hide internal submodule noise from pdoc / pydoc
__pdoc__ = {
    "chipseq_utils": False,
    "chipseq": False,
    "cython_utils": False,
    "plot_classes": False,
    "utils": False,
    "statistics": False,
    "regressionplot": False,
    "data": False,
    "igraph_classes": False,
    "_adjustText": False,
    "schemas": False,
}
