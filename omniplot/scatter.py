"""
Scatter plot functions
"""
from __future__ import annotations
from typing import Union, Optional, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.lines import Line2D

# from scipy.cluster.hierarchy import leaves_list
# from scipy.cluster import hierarchy
# from collections import defaultdict
# from natsort import natsort_keygen
# from matplotlib.patches import Rectangle
import scipy.cluster.hierarchy as sch

# import fastcluster as fcl
# from matplotlib.patches import Patch
# import sys
import matplotlib as mpl
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA, NMF, LatentDirichletAllocation

# from scipy.stats import fisher_exact
from scipy.stats import zscore
from itertools import combinations

# import os

# script_dir = os.path.dirname( __file__ )
# sys.path.append( script_dir )
import scipy.stats as stats

# from joblib import Parallel, delayed
# import itertools as it
from matplotlib.ticker import StrMethodFormatter
import statsmodels.api as sm
from sklearn.linear_model import RANSACRegressor
import copy

# from sklearn import metrics
from omniplot.utils import (
    _save,
    _dendrogram_threshold,
    _get_cluster_classes,
    # _get_cluster_classes2
    _create_color_markerlut,
    _separate_data,
    _draw_ci_pi,
    marker_list,
    _get_embedding,
    _calc_curveture,
    _baumkuchen_xy,
    _calc_r2,
    _ci_pi,
)
from omniplot._adjustText import adjust_text

# from omniplot.chipseq_utils import _calc_pearson
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.collections import PathCollection

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["svg.fonttype"] = "none"
sns.set_theme(font="Arial")

__all__ = [
    "clusterplot",
    "decomplot",
    "pie_scatter",
    "manifoldplot",
    "regression_single",
    "scatterplot",
    "volcanoplot",
]


def _scatter(
    _df: pd.DataFrame,
    x: str,
    y: str,
    cat: str,
    ax: plt.Axes,
    lut: dict[str, dict[str, dict]],
    barrierfree: bool,
    size: float | int | np.ndarray,
    legend: bool = True,
    axlabel: bool = True,
    alpha: float = 1.0,
    edgecolors: str = "w",
    linewidths: float = 1.0,
    outside: bool = False,
    legendx: float = 1.01,
    legendy: float = 1,
    c: tuple = (),
) -> PathCollection:
    if len(c) != 0:
        sc = ax.scatter(_df[x], _df[y], c=c, s=size, edgecolors=edgecolors)
        plt.colorbar(sc, ax=ax, label=cat, shrink=0.3, aspect=5, orientation="vertical")
    elif _df[cat].dtype == float:
        sc = ax.scatter(_df[x], _df[y], c=_df[cat], s=size, edgecolors=edgecolors)
        plt.colorbar(sc, ax=ax, label=cat, shrink=0.3, aspect=5, orientation="vertical")
    elif barrierfree is True:

        for key in lut[cat]["colorlut"].keys():
            _dfnew = _df.loc[_df[cat] == key]
            if isinstance(size, np.ndarray):
                _size = size[_df[cat] == key]
                sc = ax.scatter(
                    _dfnew[x],
                    _dfnew[y],
                    c=lut[cat]["colorlut"][key],
                    marker=lut[cat]["markerlut"][key],
                    label=key,
                    s=_size,
                    alpha=alpha,
                    edgecolors=edgecolors,
                    linewidths=linewidths,
                )

            else:
                sc = ax.scatter(
                    _dfnew[x],
                    _dfnew[y],
                    c=lut[cat]["colorlut"][key],
                    marker=lut[cat]["markerlut"][key],
                    label=key,
                    s=size,
                    alpha=alpha,
                    edgecolors=edgecolors,
                    linewidths=linewidths,
                )

    else:
        for key in lut[cat]["colorlut"].keys():
            _dfnew = _df.loc[_df[cat] == key]
            if type(size) == float or type(size) == int:
                sc = ax.scatter(
                    _dfnew[x],
                    _dfnew[y],
                    color=lut[cat]["colorlut"][key],
                    label=key,
                    s=size,
                    alpha=alpha,
                    edgecolors=edgecolors,
                    linewidths=linewidths,
                )

            else:
                _size = size[_df[cat] == key]
                sc = ax.scatter(
                    _dfnew[x],
                    _dfnew[y],
                    color=lut[cat]["colorlut"][key],
                    label=key,
                    s=_size,
                    alpha=alpha,
                    edgecolors=edgecolors,
                    linewidths=linewidths,
                )

    if legend is True:
        if barrierfree is True:
            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    marker=lut[cat]["markerlut"][k],
                    linewidth=0,
                    markeredgecolor=v,
                    label=k,
                    markerfacecolor=v,
                    markersize=10,
                )
                for k, v in lut[cat]["colorlut"].items()
            ]
        else:
            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linewidth=0,
                    markeredgecolor=edgecolors,
                    label=k,
                    markerfacecolor=v,
                    markersize=10,
                )
                for k, v in lut[cat]["colorlut"].items()
            ]
        if outside is True:
            ax.add_artist(
                ax.legend(
                    handles=legend_elements,
                    title=cat,
                    bbox_to_anchor=(legendx, legendy),
                )
            )
        else:
            ax.add_artist(ax.legend(handles=legend_elements, title=cat))

    if axlabel is True:
        ax.set_xlabel(x)
        ax.set_ylabel(y)
    return sc


def _add_labels(ax, df, x, y, val, topn):
    if topn > 0:
        srtindex = np.argsort(df[val])[::-1][:topn]
        _labels = np.array(df.index)
        for _label, _x, _y, _val in zip(
            _labels[srtindex],
            df[x].values[srtindex],
            df[y].values[srtindex],
            df[val].values[srtindex],
        ):
            ax.text(_x, _y, _label)

    else:
        for _label, _x, _y in zip(df.index, df[x], df[y]):
            ax.text(_x, _y, _label)


def _set_axis_format(ax, xformat, yformat, xunit, yunit, logscalex, logscaley):
    if xformat != "":

        ax.xaxis.set_major_formatter(StrMethodFormatter(xformat))
    if yformat != "":
        ax.yaxis.set_major_formatter(StrMethodFormatter(yformat))
    if yunit != "":
        ax.text(0, 1, "({})".format(yunit), transform=ax.transAxes, ha="right")
    if xunit != "":
        ax.text(1, 0, "({})".format(xunit), transform=ax.transAxes, ha="left", va="top")
    if logscalex is True:
        ax.set_xscale("log")
    if logscaley is True:
        ax.set_yscale("log")


def _marginal_plot(
    fig: Figure,
    ax: plt.Axes,
    df: pd.DataFrame,
    x: str,
    y: str,
    cat: str,
    lut: dict[str, dict[str, dict]],
    _xrange,
    _yrange,
    marginal_proportion,
):
    bb = ax.get_position()
    _x, _y, _w, _h = bb.bounds
    _ws, _hs = _w * marginal_proportion, _h * marginal_proportion
    ax.set_position([_x, _y, _ws, _hs])
    _ax = fig.add_axes([_x, _y + _hs, _ws, _h * (1 - marginal_proportion)])
    _ax.set_zorder(0)

    if cat != "":
        for key in lut[cat]["colorlut"].keys():
            _catx = df[x][df[cat] == key].values
            kernel = stats.gaussian_kde(_catx)
            _ax.fill(
                np.concatenate([[_xrange[0]], _xrange, [_xrange[-1]]]),
                np.concatenate([[0], kernel(_xrange), [0]]),
                color=lut[cat]["colorlut"][key],
                alpha=0.5,
            )

    else:
        _catx = df[x].values
        kernel = stats.gaussian_kde(_catx)
        _ax.fill(
            np.concatenate([[_xrange[0]], _xrange, [_xrange[-1]]]),
            np.concatenate([[0], kernel(_xrange), [0]]),
            color="b",
            alpha=0.5,
        )

    _ax.set_xticks([])
    _ax = fig.add_axes([_x + _ws, _y, _w * (1 - marginal_proportion), _hs])
    _ax.set_zorder(0)
    if cat != "":
        for key in lut[cat]["colorlut"].keys():
            _catx = df[y][df[cat] == key].values
            kernel = stats.gaussian_kde(_catx)
            _ax.fill(
                np.concatenate([[0], kernel(_yrange), [0]]),
                np.concatenate([[_yrange[0]], _yrange, [_yrange[-1]]]),
                color=lut[cat]["colorlut"][key],
                alpha=0.5,
            )
    else:
        _catx = df[y].values
        kernel = stats.gaussian_kde(_catx)
        _ax.fill(
            np.concatenate([[0], kernel(_yrange), [0]]),
            np.concatenate([[_yrange[0]], _yrange, [_yrange[-1]]]),
            color="b",
            alpha=0.5,
        )
    _ax.set_yticks([])


def _scale_size(x, size_scale, smin, smax):
    return size_scale * (0.01 + (x - smin) / (smax - smin))


def _reverse_size(x, size_scale, smin, smax):
    return (x / size_scale - 0.01) * (smax - smin) + smin


def scatterplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    ax: Optional[plt.Axes] = None,
    fig: Figure | None = None,
    colors: str | list[str] = "",
    category: Union[str, List[str]] = "",
    sizes: str = "",
    marginal_dist: bool = False,
    kde: bool = False,
    kde_kw: dict = {},
    kmeans: bool = False,
    n_clusters: int = 3,
    cluster_center: bool = True,
    kmeans_kw: dict = {},
    regression: bool = False,
    robust_param: dict = {},
    regression_color: str = "lightgreen",
    c: list | np.ndarray | None = None,
    cname: str = "",
    size_scale: float = 100,
    palette: str = "",
    palette_cat: str | dict = "tab20c",
    palette_val: str = "coolwarm",
    size_legend_num: int = 4,
    markers: bool = False,
    size: float = 30.0,
    show_labels: dict | None = None,
    alpha: float = 1,
    edgecolors: str = "w",
    linewidths: float = 1,
    cbar_format: str = "",
    size_format: str = "",
    xformat: str = "",
    yformat: str = "",
    xunit: str = "",
    yunit: str = "",
    size_unit: str = "",
    color_unit: str = "",
    color: str = "b",
    axlabel: str = "single",
    title: str = "",
    show_legend: bool = True,
    logscalex: bool = False,
    logscaley: bool = False,
    figsize: list | None = None,
    rows_cols: list | None = None,
    save: str = "",
    gridspec_kw: dict | None = None,
) -> Dict:
    """
    Simple scatter plot. almost same function with seaborn.scatterplot.

    Parameters
    ----------
    df : pandas DataFrame


    x, y: str, optional
        The column names to be the x and y axes of scatter plots. If reduce_dimension=True, these options will be
        ignored.
    colors: Union[str, list]="", optional
        The names of columns (containing numerial values) to appear as a gradient of point colors.
    category: Union[str, list]="", optional
        The names of columns (containing categorical values) to appear as color labels
    sizes: str="", optional
        The names of columns (containing numerial values) to appear as point sizes.
    marginal_dist: bool, optional (default: False)
        Whether to draw marginal distributions
    kde: bool, optional (default: False)
        Whether to overlay KDE plot.
    kde_kw: dict, optional
        KDE plot keyword arguements. See https://seaborn.pydata.org/generated/seaborn.kdeplot.html for details.

    kmeans: bool, optional (default: False)
        Whether to calculate and draw kmean clusters
    n_clusters: int, optional (default: 3)
        K-means cluster number.
    cluster_center: bool, optional (default: True)
        Whether to draw lines from the k-means centers.https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html for details
    kmeans_kw: dict,
        KMeans keyword arguments. See
    c: Union[List, np.ndarray], optional
        1 dimensional array containing values shown as point facecolors or 2 dimensional array consists of RGB(A)
    cname: str, optional
        Label for the color value specified by "c" option
    palette: str="", optional

    palette_cat: str, optional (default: "tab20c")
        The color palette for categorical labels
    palette_val: str, optional (default: "coolwarm")
        The color palette for the color gradient specified by the "colors" option.
    show_labels: dict, optional
        A dictionary to specify the condition to add labels to the points. dataframe index will be labeled to points.
        It may contain "val" and "topn" keys. if you want all points to labeled, pass {"topn":0} to the option.
        To add labels to the only top n points of specific values, pass a dictionary like {"val": "body_mass_g", "topn":5}.
        "val" specify the column name to rank points and "topn" specify the number of points to be labeled.

    save: str, optional
        Prefix to save the figure
    title: str, optional
        The title for the figure.
    markers: bool, optional (default: False)
        Whether to use markers to different categorical labels
    rows_cols: list, optional
        The number of rows and columns for subplots
    size: float, optional (default: 10)
        point size. This will be ignored when "sizes" option is set.
    xunit: str, optional
        The x axis unit.
    yunit:str, optional
        The y axis unit.
    size_unit: str, optional
        The unit of point sizes when "sizes" option is set.
    color: str, optional
    axlabel: str, optional (default:"single"), ["single", "each", "non"]
        How to show the labels of the axes. "each" will add the labels to each axis of subplots. "single" will add single axis labels to the figure.
    logscalex: bool=False,
        Whether to transform the x axis to the log scale.
    logscaley: bool=False,
        Whether to transform the y axis to the log scale.
    figsize: list=[], optional
        figure size. if not set, it automatically calculate a reasonable figure size.
    alpha: float=1,
        Alpha of points.
    size_scale: float=60,
        The scale of the point sizes, only effective when "sizes" option is set. If the sizes of points are too small or too large, adjust the sizes with this option.
    edgecolors: str="w", optional
        The point edge color.
    linewidths: float=1, optional
        The point edge width.
    cbar_format, size_format, xformat, yformat: str
        e.g., "{x:.2f}", '{x:.3E}'

    gridspec_kw: dict, optional
        Parameters for matplotlib gridspec. (https://matplotlib.org/stable/api/_as_gen/matplotlib.gridspec.GridSpec.html)
        e.g., {"wspace":0.75,"hspace":0.5}
    adjust_kw: dict, optional

    Return
    ------
    {"axes":axes, "fig":fig,
    "regression_models":regression_models, "regression_results":regression_results,
    "kmeans_result":_kmeans, "df":df} : dict

    """
    # functions to scale and rescale the size of each point

    if len(kde_kw) == 0:
        kde_kw = dict(alpha=0.5, levels=4)

    if palette != "":
        palette_cat = palette
        palette_val = palette

    # original_index = df.index

    X, category = _separate_data(df, variables=[x, y], category=category)
    if len(colors) != 0:
        if type(colors) == str:
            colors = [colors]
    else:
        colors = []

    if c is None:
        c = []
    c = np.array(c)
    if len(c) != 0:
        if cname == "":
            cname = "c"
        if len(c.shape) == 1:
            df[cname] = c
            colors.append(cname)

    # Calculating clusters and add cluster labels to dataframe
    _kmeans = None
    if kmeans is True:
        _kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(
            X, *kmeans_kw
        )
        df["kmeans"] = _kmeans.labels_
        _kmeanlabels = np.unique(_kmeans.labels_)
        category.append("kmeans")

    totalnum = len(category) + len(colors) + int(len(c.shape) == 2)
    if totalnum <= 1:
        totalnum = 1
        axlabel = "each"
    # determining the figure size and the number of rows and columns.
    gridspec_kw = _get_gridspec_kw(gridspec_kw, show_legend, totalnum, regression)
    fig, axes = _get_axes(
        fig, ax, totalnum, gridspec_kw, marginal_dist, rows_cols, regression
    )

    if axlabel == "single":
        _axlabeleach = False
    elif axlabel == "non":
        _axlabeleach = False
    elif axlabel == "each":
        _axlabeleach = True

    size_legend_elements = get_point_sizes(sizes, df, size_scale, size_legend_num)

    legendx = 1.01
    legendy = 1

    # Preparing x and y ranges for marginal distribution.
    margins = 0.1
    if marginal_dist is True:

        _xmin, _xmax = np.amin(df[x]), np.amax(df[x])
        _xmargin = margins * (_xmax - _xmin)
        _ymin, _ymax = np.amin(df[y]), np.amax(df[y])
        _ymargin = margins * (_ymax - _ymin)
        _xrange = np.linspace(_xmin - _xmargin, _xmax + _xmargin, 1000)
        _yrange = np.linspace(_ymin - _ymargin, _ymax + _ymargin, 1000)
        marginal_proportion = 0.8
        legendx = legendx / marginal_proportion

    i = 0
    # Drawing scatter plots
    lut = {}
    regression_models = {}
    regression_results = {}
    if len(category) != 0:

        for cat in category:
            ax = axes[i]
            ax.margins(margins)
            ax.set_zorder(1)
            i += 1

            _clut, _mlut = _create_color_markerlut(df, cat, palette_cat, marker_list)

            lut[cat] = {"colorlut": _clut, "markerlut": _mlut}

            # Plotting KMeans results
            if cat == "kmeans" and cluster_center is True and regression is False:
                for ul, center in zip(_kmeanlabels, _kmeans.cluster_centers_):
                    _df = df.loc[df["kmeans"] == ul]
                    for _x, _y in zip(_df[x], _df[y]):
                        ax.plot(
                            [center[0], _x],
                            [center[1], _y],
                            color=_clut[ul],
                            alpha=0.25,
                        )

            # Plotting regression results
            if regression is True:
                fitted_models, reg_results = _regression(
                    df, x, y, ax, cat=cat, _clut=_clut, robust_param=robust_param
                )
                regression_models.update(fitted_models)
                regression_results.update(reg_results)

            # Plotting a scatter plot
            sc = _scatter(
                df,
                x,
                y,
                cat,
                ax,
                lut,
                markers,
                size,
                axlabel=_axlabeleach,
                alpha=alpha,
                edgecolors=edgecolors,
                linewidths=linewidths,
                outside=True,
                legendx=legendx,
                legendy=legendy,
                legend=show_legend,
            )

            # Plotting a KDE plot
            if kde is True:
                sns.kdeplot(
                    data=df,
                    x=x,
                    y=y,
                    hue=cat,
                    ax=ax,
                    palette=_clut,
                    legend=False,
                    **kde_kw,
                )
                ax.set(xlabel=None)
                ax.set(ylabel=None)

            # Plotting marginal distribution
            if marginal_dist is True:
                _marginal_plot(
                    fig, ax, df, x, y, cat, lut, _xrange, _yrange, marginal_proportion
                )

            # Setting the format of axes
            _set_axis_format(ax, xformat, yformat, xunit, yunit, logscalex, logscaley)

            # Drawing the legend of point sizes
            if sizes != "" and show_legend is True:

                if size_unit != "":
                    sizes = sizes + "(" + size_unit + ")"
                ax.add_artist(
                    ax.legend(
                        handles=size_legend_elements,
                        title=sizes,
                        bbox_to_anchor=(legendx, 0.6),
                    )
                )

            # Drawing point labels
            if isinstance(show_labels, dict) and "topn" in colors:
                _add_labels(ax, df, x, y, show_labels["val"], show_labels["topn"])

    # Plotting scatter plots with color values specified by "colors"
    if len(colors) != 0:
        if type(color_unit) == str:
            color_unit = [color_unit]

        for _c, _unit in zip(colors, color_unit):
            ax = axes[i]
            ax.margins(margins)
            ax.set_zorder(1)
            i += 1
            _df = df.sort_values(by=[_c], ascending=True)
            if type(size) == float or type(size) == int:
                _size = size
            else:
                _size = size[np.argsort(df[_c])]

            if regression is True:
                fitted_models, reg_results = _regression(
                    df,
                    x,
                    y,
                    ax,
                    robust_param=robust_param,
                    newkey="total",
                    color=regression_color,
                )
                regression_models.update(fitted_models)
                regression_results.update(reg_results)

            sc = ax.scatter(
                _df[x],
                _df[y],
                c=_df[_c],
                cmap=palette_val,
                s=_size,
                alpha=alpha,
                edgecolors=edgecolors,
                linewidths=linewidths,
            )
            if kde is True:
                sns.kdeplot(
                    data=df, x=x, y=y, ax=ax, color=color, legend=False, **kde_kw
                )
                ax.set(xlabel=None)
                ax.set(ylabel=None)

            if marginal_dist is True:
                _marginal_plot(
                    fig, ax, df, x, y, "", lut, _xrange, _yrange, marginal_proportion
                )

            bb = ax.get_position()
            axx, axy, axw, axh = bb.bounds
            _xcax = axx + axw * 1.005
            if marginal_dist is True:
                _xcax = axx + axw * 1.005 / marginal_proportion
            cax = plt.axes([_xcax, axy, 0.02, 0.1])
            if _unit != "":
                _c = _c + "({})".format(_unit)
            plt.colorbar(
                sc,
                cax=cax,
                label=_c,
                shrink=0.3,
                aspect=5,
                orientation="vertical",
                anchor=(0.2, 0),
            )
            if cbar_format != "":
                ax.xaxis.set_major_formatter(StrMethodFormatter(cbar_format))

            # ax.set_title(_c)
            if _axlabeleach is True:
                ax.set_xlabel(x)
                ax.set_ylabel(y)

            _set_axis_format(ax, xformat, yformat, xunit, yunit, logscalex, logscaley)

            if sizes != "" and show_legend is True:
                if size_unit != "":
                    sizes = sizes + "(" + size_unit + ")"
                ax.add_artist(
                    ax.legend(
                        handles=size_legend_elements,
                        title=sizes,
                        bbox_to_anchor=(legendx, 1),
                    )
                )

            if show_labels is not None and "topn" in show_labels:
                _add_labels(ax, df, x, y, show_labels["val"], show_labels["topn"])

    if int(len(c.shape) > 1):
        ax = axes[i]
        i += 1
        if type(color_unit) == str:
            color_unit = [color_unit]
        _unit = color_unit[-1]
        ax.margins(margins)
        ax.set_zorder(1)

        if regression is True:
            fitted_models, reg_results = _regression(
                df,
                x,
                y,
                ax,
                robust_param=robust_param,
                newkey="total",
                color=regression_color,
            )
            regression_models.update(fitted_models)
            regression_results.update(reg_results)

        sc = ax.scatter(
            df[x], df[y], c=c, s=size, edgecolors=edgecolors, linewidths=linewidths
        )
        ax.text(
            0.1,
            0.8,
            cname,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1, alpha=0.8),
        )

        if kde is True:
            sns.kdeplot(data=df, x=x, y=y, ax=ax, color=color, legend=False, **kde_kw)
            ax.set(xlabel=None)
            ax.set(ylabel=None)

        if marginal_dist is True:
            _marginal_plot(
                fig, ax, df, x, y, "", lut, _xrange, _yrange, marginal_proportion
            )

        if _axlabeleach is True:
            ax.set_xlabel(x)
            ax.set_ylabel(y)

        _set_axis_format(ax, xformat, yformat, xunit, yunit, logscalex, logscaley)

        if sizes != "" and show_legend is True:
            if size_unit != "":
                sizes = sizes + "(" + size_unit + ")"
            ax.add_artist(
                ax.legend(
                    handles=size_legend_elements,
                    title=sizes,
                    bbox_to_anchor=(legendx, 1),
                )
            )

        if len(show_labels) != 0:
            _add_labels(ax, df, x, y, show_labels["val"], show_labels["topn"])

    if len(category) + len(colors) == 0 and int(len(c.shape) != 2):

        ax = axes[i]
        if regression is True:
            fitted_models, reg_results = _regression(
                df,
                x,
                y,
                ax,
                robust_param=robust_param,
                newkey="total",
                color=regression_color,
            )
            regression_models.update(fitted_models)
            regression_results.update(reg_results)

        sc = ax.scatter(
            df[x],
            df[y],
            c=color,
            s=size,
            alpha=alpha,
            edgecolors=edgecolors,
            linewidths=linewidths,
        )
        if kde is True:
            sns.kdeplot(data=df, x=x, y=y, ax=ax, color=color, **kde_kw)
            ax.set(xlabel=None)
            ax.set(ylabel=None)

        ax.set_xlabel(x)
        ax.set_ylabel(y)

        _set_axis_format(ax, xformat, yformat, xunit, yunit, logscalex, logscaley)

        if sizes != "" and show_legend is True:
            if size_unit != "":
                sizes = sizes + "(" + size_unit + ")"
            ax.add_artist(
                ax.legend(
                    handles=size_legend_elements, title=sizes, bbox_to_anchor=(1.01, 1)
                )
            )

    if title != "":
        if fig is None:
            fig.suptitle(title)
        else:
            plt.title(title)

    if axlabel == "single" and fig is None:
        bbox = axes[0].get_position()
        fig.text(0.5, 0.05, x, ha="center", fontsize="large")
        fig.text(
            bbox.bounds[0] * 0.5,
            0.5,
            y,
            va="center",
            rotation="vertical",
            fontsize="large",
        )

    if len(axes) != totalnum:
        for i in range(len(axes) - totalnum):
            axes[-(i + 1)].set_axis_off()

    _save(save, "scatter")
    #  plt.tight_layout()

    return {
        "axes": axes,
        "fig": fig,
        "regression_models": regression_models,
        "regression_results": regression_results,
        "kmeans_result": _kmeans,
        "df": df,
    }


def _get_gridspec_kw(gridspec_kw, show_legend, totalnum, regression):
    if gridspec_kw:
        return gridspec_kw
    _right = 0
    if show_legend is False:
        _right += 0.16
    if totalnum == 1:
        if regression is True:
            gridspec_kw = {"right": 0.67 + _right, "bottom": 0.3}
        else:
            gridspec_kw = {"right": 0.67 + _right, "bottom": 0.15}
    elif totalnum == 2:
        if regression is True:
            gridspec_kw = {
                "wspace": 0.75,
                "hspace": 0.5,
                "right": 0.85,
                "bottom": 0.35,
                "top": 0.95,
            }
        else:
            gridspec_kw = {"wspace": 0.75, "right": 0.85, "top": 0.95}
    else:
        if regression is True:
            gridspec_kw = {
                "wspace": 0.75,
                "hspace": 0.5,
                "right": 0.85,
                "bottom": 0.2,
                "top": 0.95,
            }
        else:
            gridspec_kw = {"wspace": 0.75, "right": 0.85, "top": 0.95}
    return gridspec_kw


def get_point_sizes(sizes, df, size_scale, size_legend_num, size_format: str = ""):
    # Creating point size array protional to values in the column specified by "sizes" option.
    # Point sizes are scaled maximum to be size_scale (in order to avoid too small/large points).
    # And creating a size legend of which size labels correspond to the original values
    size_legend_elements = []
    if sizes == "":
        return size_legend_elements

    size = df[sizes]
    size = np.nan_to_num(size)
    smin = np.amin(size)
    smax = np.amax(size)
    size = _scale_size(size, size_scale, smin, smax)

    vmin, vmax = np.min(size), np.max(size)
    vinterval = (vmax - vmin) / (size_legend_num - 1)
    size_legend_elements = []
    if size_format == "":
        if 1 < np.abs(vmax) <= 1000:
            size_format = "{x:.2f}"
        elif 0 < np.abs(vmax) <= 1 or 1000 < np.abs(vmax):
            size_format = "{x:.3E}"
    for _i in range(size_legend_num):
        s = vmin + _i * vinterval
        _s = _reverse_size(s, size_scale, smin, smax)
        size_legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linewidth=0,
                markeredgecolor="white",
                markersize=s ** (0.5),
                label=size_format.format(x=_s),
                markerfacecolor="black",
            )
        )
    return size_legend_elements


def _get_axes(
    fig, ax, totalnum, gridspec_kw, marginal_dist, rows_cols, regression, figsize=None
):
    if ax is not None:
        if totalnum == 1 and type(ax) == plt.Axes:

            axes = [ax]
        elif totalnum > 1 and type(ax) == np.ndarray:
            axes = ax.flatten()
        elif totalnum > 1 and type(ax) == list:
            axes = ax
        elif totalnum > 1 and ax == plt.Axes:
            raise ValueError(
                "If you provide the ax option, the number of plots and axes must be equal. \
                            The total number of plots will be the sum of category and colors plus whether or not c and kmeans options provided"
            )
        plt.subplots_adjust(**gridspec_kw)
        totalnum = 1
        if marginal_dist is True and fig is None:
            raise ValueError(
                "if you pass an axis opject and want to draw marginal distribution, you also need to give a figure object"
            )

    elif rows_cols is None or len(rows_cols) == 0:

        if totalnum <= 1:
            if figsize is None:
                figsize = [6, 4]
            fig, ax = plt.subplots(figsize=figsize, gridspec_kw=gridspec_kw)
            axes = [ax]
        else:
            if figsize is None:
                if regression is True:
                    figsize = [9, 5 * totalnum // 2 + int(totalnum % 2 != 0)]
                else:
                    figsize = [9, 4 * totalnum // 2 + int(totalnum % 2 != 0)]

            fig, axes = plt.subplots(
                nrows=totalnum // 2 + int(totalnum % 2 != 0),
                ncols=2,
                figsize=figsize,
                gridspec_kw=gridspec_kw,
            )
            axes = axes.flatten()
    else:
        if figsize is None:
            figsize = [10, 4 * totalnum // 2 + int(totalnum % 2 != 0)]
        fig, axes = plt.subplots(
            nrows=rows_cols[0],
            ncols=rows_cols[1],
            figsize=figsize,
            gridspec_kw=gridspec_kw,
        )
        axes = axes.flatten()

    return fig, axes


def clusterplot(
    df: pd.DataFrame,
    variables: List = [],
    category: Union[List[str], str] = "",
    method: str = "kmeans",
    n_clusters: Union[str, int] = 3,
    x: str = "",
    y: str = "",
    size: float = 10,
    reduce_dimension: str = "umap",
    testrange: list = [1, 20],
    topn_cluster_num: int = 2,
    show: bool = False,
    min_dist: float = 0.25,
    n_neighbors: int = 15,
    eps: Union[List[float], float] = 0.5,
    pcacomponent: Optional[int] = None,
    ztranform: bool = True,
    palette: list = ["Spectral", "tab20b"],
    save: str = "",
    title: str = "",
    markers: bool = False,
    ax: Optional[plt.Axes] = None,
    piesize_scale: float = 0.02,
    min_cluster_size: int = 10,
    **kwargs,
) -> Dict:
    """
    Clustering data and draw them as a scatter plot optionally with dimensionality reduction.

    Parameters
    ----------
    df : pandas DataFrame
    x, y: str, optional
        The column names to be the x and y axes of scatter plots. If reduce_dimension=True, these options will be
        ignored.

    variables: list, optional
        The names of variables to calculate clusters..

    category: str, optional
        the column name of a known sample category (if exists).
    method: str
        Method name for clustering.
        "kmeans"
        "hierarchical",
        "dbscan": Density-Based Clustering Algorithms
        "fuzzy" : fuzzy c-mean clustering using scikit-fuzzy
    n_clusters: int or str, optional (default: 3)
        The number of clusters to be created. If "auto" is provided, it will estimate optimal
        cluster numbers with "Sum of squared distances" for k-mean clustering and silhouette method for others.
    eps: int or list[int]
        DBSCAN's hyper parameter. It will affect the total number of clusters.
    reduce_dimension: str, optional (default: "umap")
        Dimensionality reduction method. if "" is passed, no reduction methods are applied.
        In this case, data must have only two dimentions or x and y options must be specified.

    markers: bool, optional (default: False)
        Whether to use different markers for each cluster/category (for a colorblind-friendly plot).
    show: bool, optional (default: False)
        Whether to show figures
    size: float, optional (default: 10)
        The size of points in the scatter plot.

    testrange: list, optional (default: [1,20])
        The range of cluster numbers to be tested when n_clusters="auto".
    topn_cluster_num: int, optional (default: 2)
        Top n optimal cluster numbers to be plotted when n_clusters="auto".


    min_dist: float, optional (default: 0.25)
        A UMAP parameter
    n_neighbors: int, optinal (default: 15)
        A UMAP parameter.
    eps: Union[List[float], float], optional (default: 0.5)
        A DBSCAN parameter.
    pcacomponent: Optional[int]=None,
        The number of PCA component. PCA result will be used by UMAP and hierarchical clustering.
    ztranform: bool, optinal (default: True)
        Whether to convert data into z scores.
    palette: list, optional (default: ["Spectral","cubehelix"])

    save: str, optional

    piesize_scale: float=0.02
    Returns
    -------
    Raises
    ------
    Notes
    -----
    References
    ----------
    See Also
    --------
    Examples
    --------
    """

    original_index = df.index

    X, category = _separate_data(df, variables=variables, category=category)

    if ztranform is True:
        X = zscore(X, axis=0)

    if pcacomponent is None:

        if 20 < X.shape[1]:
            pcacomponent = 20
        elif 10 < X.shape[1]:
            pcacomponent = 10
        else:
            pcacomponent = 2
    pca = PCA(n_components=pcacomponent, random_state=1)
    xpca = pca.fit_transform(X)

    if reduce_dimension == "umap":
        import umap

        u = umap.UMAP(random_state=42, min_dist=min_dist, n_neighbors=n_neighbors)
        X = u.fit_transform(xpca)
        x = "UMAP1"
        y = "UMAP2"
    elif reduce_dimension == "tsne":
        tsne = _get_embedding("tsne")
        # u=umap.UMAP(random_state=42, min_dist=min_dist,n_neighbors=n_neighbors)
        X = tsne.fit_transform(xpca)
        x = "TSNE1"
        y = "TSNE2"
    if n_clusters == "auto" and method == "kmeans":
        Sum_of_squared_distances = []
        K = list(range(*testrange))
        for k in K:
            km = KMeans(n_clusters=k, n_init=10)
            km = km.fit(X)
            Sum_of_squared_distances.append(km.inertia_)
        normy = np.array(Sum_of_squared_distances) / np.amax(Sum_of_squared_distances)
        normy = 1 - normy
        normx = np.linspace(0, 1, len(K))
        perp = _calc_curveture(normx, normy)
        srtindex = np.argsort(perp)[::-1]
        plt.subplots()
        plt.plot(K, Sum_of_squared_distances, "-", label="Sum of squared distances")
        plt.plot(K, perp * np.amax(Sum_of_squared_distances), label="curveture")
        for i in range(topn_cluster_num):
            plt.plot(
                [K[srtindex[i]], K[srtindex[i]]],
                [0, np.amax(Sum_of_squared_distances)],
                "--",
                color="r",
            )
            plt.text(
                K[srtindex[i]],
                np.amax(Sum_of_squared_distances) * 0.95,
                "N=" + str(K[srtindex[i]]),
            )
        plt.xticks(K)
        plt.xlabel("Cluster number")
        plt.ylabel("Sum of squared distances")
        plt.title("Elbow method for optimal cluster number")
        plt.legend()
        # print("Top two optimal cluster No are: {}, {}".format(K[srtindex[0]],K[srtindex[1]]))
        # n_clusters=[K[srtindex[0]],K[srtindex[1]]]
        n_clusters = [K[i] for i in srtindex[:topn_cluster_num]]
        print("Top two optimal cluster No are:", n_clusters)
        _save(save, method)
    elif n_clusters == "auto" and method == "fuzzy":
        try:
            import skfuzzy as fuzz
        except ImportError:
            from pip._internal import main as pip

            pip(["install", "--user", "scikit-fuzzy"])
            import skfuzzy as fuzz
        fpcs = []
        K = list(range(*testrange))
        _X = X.T
        for nc in K:

            cntr, u, u0, d, jm, p, fpc = fuzz.cmeans(
                _X, nc, 2, error=0.005, maxiter=1000, init=None
            )

            fpcs.append(fpc)

        srtindex = np.argsort(fpcs)[::-1]
        plt.subplots()
        plt.plot(K, fpcs, "-")

        for i in range(topn_cluster_num):
            plt.plot(
                [K[srtindex[i]], K[srtindex[i]]], [0, np.amax(fpcs)], "--", color="r"
            )
            plt.text(K[srtindex[i]], np.amax(fpcs) * 0.95, "N=" + str(K[srtindex[i]]))
        plt.xticks(K)
        plt.xlabel("Cluster number")
        plt.ylabel("Fuzzy partition coefficient")
        n_clusters = [K[i] for i in srtindex[:topn_cluster_num]]
        print("Top two optimal cluster No are:", n_clusters)

        _save(save, method)
    elif n_clusters == "auto" and method == "hierarchical":
        import scipy.spatial.distance as ssd

        labels = df.index
        D = ssd.squareform(ssd.pdist(xpca))
        Y = sch.linkage(D, method="ward")
        Z = sch.dendrogram(Y, labels=labels, no_plot=True)

        K = list(range(*testrange))
        newK = []
        scores = []
        for k in K:
            t = _dendrogram_threshold(Z, k)
            Z2 = sch.dendrogram(Y, labels=labels, color_threshold=t, no_plot=True)
            clusters = _get_cluster_classes(Z2, label="ivl")
            _k = len(clusters)
            if _k not in newK:
                newK.append(_k)
                sample2cluster = {}
                i = 1
                for k, v in clusters.items():
                    for sample in v:
                        sample2cluster[sample] = "C" + str(i)
                    i += 1
                scores.append(
                    silhouette_score(
                        X,
                        [sample2cluster[sample] for sample in labels],
                        metric="euclidean",
                    )
                    / _k
                )

        scores = np.array(scores)
        srtindex = np.argsort(scores)[::-1]
        plt.subplots()
        plt.plot(newK, scores, "-")
        for i in range(topn_cluster_num):
            plt.plot(
                [newK[srtindex[i]], newK[srtindex[i]]],
                [0, np.amax(scores)],
                "--",
                color="r",
            )
            plt.text(
                newK[srtindex[i]], np.amax(scores) * 0.95, "N=" + str(newK[srtindex[i]])
            )
        plt.xticks(newK)
        plt.xlabel("Cluster number")
        plt.ylabel("Silhouette scores")
        plt.title("Optimal cluster number searches by silhouette method")

        n_clusters = [newK[i] for i in srtindex[:topn_cluster_num]]
        print("Top two optimal cluster No are:", n_clusters)
        _save(save, method)
    elif n_clusters == "auto" and method == "dbscan":

        from sklearn.neighbors import NearestNeighbors

        neigh = NearestNeighbors(n_neighbors=2)
        nbrs = neigh.fit(X)
        distances, indices = nbrs.kneighbors(X)
        distances = np.sort(distances[:, 1], axis=0)

        K = np.linspace(np.amin(distances), np.amax(distances), 20)
        newK = []
        scores = []
        _K = []
        for k in K:
            db = DBSCAN(eps=k, min_samples=5, n_jobs=-1)
            dbX = db.fit(X)
            labels = np.unique(dbX.labels_[dbX.labels_ >= 0])

            if len(labels) < 2:
                continue
            _k = len(labels)
            if _k not in newK:
                newK.append(_k)
                _K.append(k)
                scores.append(
                    silhouette_score(
                        X[dbX.labels_ >= 0],
                        dbX.labels_[dbX.labels_ >= 0],
                        metric="euclidean",
                    )
                    / _k
                )

        scores = np.array(scores)

        _ksort = np.argsort(newK)
        _K = np.array(_K)[_ksort]
        newK = np.array(newK)[_ksort]
        scores = np.array(scores)[_ksort]
        srtindex = np.argsort(scores)[::-1]
        plt.subplots()
        plt.plot(newK, scores, "-")

        for i in range(topn_cluster_num):
            plt.plot(
                [_K[srtindex[i]], _K[srtindex[i]]],
                [0, np.amax(scores)],
                "--",
                color="r",
            )
            plt.text(
                _K[srtindex[i]], np.amax(scores) * 0.95, "N=" + str(newK[srtindex[i]])
            )
        plt.xticks(newK)
        plt.xlabel("eps")
        plt.ylabel("Silhouette scores")
        plt.title("Optimal cluster number searches by silhouette method")

        _n_clusters = [newK[i] for i in range(topn_cluster_num)]
        print("Top two optimal cluster No are:", _n_clusters)
        eps = [_K[i] for i in srtindex[:topn_cluster_num]]
        _save(save, method)

    elif n_clusters == "auto" and method == "hdbscan":
        try:
            import hdbscan
        except ImportError:
            from pip._internal import main as pip

            pip(["install", "--user", "hdbscan"])
            import hdbscan

        from sklearn.neighbors import NearestNeighbors

        neigh = NearestNeighbors(n_neighbors=2)
        nbrs = neigh.fit(X)
        distances, indices = nbrs.kneighbors(X)
        distances = np.sort(distances[:, 1], axis=0)

        # K=np.linspace(0.01,1,10)
        K = np.arange(2, 20, 1)
        print(K)
        newK = []
        scores = []
        _K = []
        for k in K:
            db = hdbscan.HDBSCAN(
                min_cluster_size=k,
                # cluster_selection_epsilon=k,
                algorithm="best",
                alpha=1.0,
                leaf_size=40,
                metric="euclidean",
                min_samples=None,
                p=None,
                core_dist_n_jobs=-1,
            )
            dbX = db.fit(X)
            labels = np.unique(dbX.labels_[dbX.labels_ >= 0])

            if len(labels) < 2:
                continue
            _k = len(labels)
            if _k not in newK:
                newK.append(_k)
                _K.append(k)
                scores.append(
                    silhouette_score(
                        X[dbX.labels_ >= 0],
                        dbX.labels_[dbX.labels_ >= 0],
                        metric="euclidean",
                    )
                    / _k
                )

        scores = np.array(scores)

        _ksort = np.argsort(newK)
        _K = np.array(_K)[_ksort]
        newK = np.array(newK)[_ksort]
        scores = np.array(scores)[_ksort]
        srtindex = np.argsort(scores)[::-1]
        plt.subplots()
        plt.plot(newK, scores, "-")

        for i in range(topn_cluster_num):
            plt.plot(
                [_K[srtindex[i]], _K[srtindex[i]]],
                [0, np.amax(scores)],
                "--",
                color="r",
            )
            plt.text(
                _K[srtindex[i]], np.amax(scores) * 0.95, "N=" + str(newK[srtindex[i]])
            )
        plt.xticks(_K)
        plt.xlabel("min_cluster_size")
        plt.ylabel("Silhouette scores")
        plt.title("Optimal cluster number searches by silhouette method")

        _n_clusters = [newK[i] for i in range(topn_cluster_num)]
        print("Top two optimal cluster No are:", _n_clusters)
        min_cluster_size = [_K[i] for i in srtindex[:topn_cluster_num]]
        _save(save, method)
    else:
        n_clusters = [n_clusters]
    if method == "kmeans":
        dfnews = []

        for nc in n_clusters:
            kmean = KMeans(n_clusters=nc, random_state=0, n_init=10)
            kmX = kmean.fit(X)
            labels = np.unique(kmX.labels_)

            dfnew = pd.DataFrame(
                data=np.array([X[:, 0], X[:, 1]]).T,
                columns=[x, y],
                index=original_index,
            )
            dfnew["kmeans"] = kmX.labels_
            dfnews.append(dfnew)
        hue = "kmeans"

    elif method == "hierarchical":
        import scipy.spatial.distance as ssd

        labels = df.index
        D = ssd.squareform(ssd.pdist(xpca))
        Y = sch.linkage(D, method="ward")
        Z = sch.dendrogram(Y, labels=labels, no_plot=True)

        dfnews = []
        for nc in n_clusters:
            t = _dendrogram_threshold(Z, nc)
            Z2 = sch.dendrogram(Y, labels=labels, color_threshold=t, no_plot=True)
            clusters = _get_cluster_classes(Z2, label="ivl")
            sample2cluster = {}
            i = 1
            for k, v in clusters.items():
                for sample in v:
                    sample2cluster[sample] = "C" + str(i)
                i += 1

            dfnew = pd.DataFrame(
                data=np.array([X[:, 0], X[:, 1]]).T,
                columns=[x, y],
                index=original_index,
            )
            dfnew["hierarchical"] = [sample2cluster[sample] for sample in labels]
            dfnews.append(dfnew)
        hue = "hierarchical"
    elif method == "dbscan":
        dfnews = []

        if type(eps) == float:
            eps = [eps]
        n_clusters = []
        for e in eps:
            db = DBSCAN(eps=e, min_samples=5, n_jobs=-1)
            dbX = db.fit(X)
            labels = np.unique(dbX.labels_)

            dfnew = pd.DataFrame(
                data=np.array([X[:, 0], X[:, 1]]).T,
                columns=[x, y],
                index=original_index,
            )
            dfnew["dbscan"] = dbX.labels_
            dfnews.append(dfnew)
            tmp = 0
            for c in set(dbX.labels_):
                if c >= 0:
                    tmp += 1
            n_clusters.append(str(tmp) + ", eps=" + str(np.round(e, 2)))

        hue = "dbscan"
    elif method == "hdbscan":
        dfnews = []

        try:
            import hdbscan
        except ImportError:
            from pip._internal import main as pip

            pip(["install", "--user", "hdbscan"])
            import hdbscan

        if type(min_cluster_size) == int:
            min_cluster_size = [min_cluster_size]
        n_clusters = []
        fuzzylabels = []
        for e in min_cluster_size:
            db = hdbscan.HDBSCAN(
                min_cluster_size=e,
                prediction_data=True,
                algorithm="best",
                alpha=1.0,
                approx_min_span_tree=True,
                gen_min_span_tree=True,
                leaf_size=40,
                metric="euclidean",
                min_samples=None,
                p=None,
            )
            dbX = db.fit(X)
            labels = np.unique(dbX.labels_)

            dfnew = pd.DataFrame(
                data=np.array([X[:, 0], X[:, 1]]).T,
                columns=[x, y],
                index=original_index,
            )
            dfnew["dbscan"] = dbX.labels_
            fuzzylabels.append(hdbscan.all_points_membership_vectors(dbX))
            dfnews.append(dfnew)
            tmp = 0
            for c in set(dbX.labels_):
                if c >= 0:
                    tmp += 1
            n_clusters.append(str(tmp) + ", eps=" + str(np.round(e, 2)))

        hue = "hdbscan"
    elif method == "fuzzy":
        try:
            import skfuzzy as fuzz
        except ImportError:
            from pip._internal import main as pip

            pip(["install", "--user", "scikit-fuzzy"])
            import skfuzzy as fuzz

        dfnews = []
        fuzzylabels = []

        _X = X.T
        for nc in n_clusters:

            cntr, u, u0, d, jm, p, fpc = fuzz.cmeans(
                _X, nc, 2, error=0.005, maxiter=1000, init=None
            )

            dfnew = pd.DataFrame(
                data=np.array([X[:, 0], X[:, 1]]).T,
                columns=[x, y],
                index=original_index,
            )
            fuzzylabels.append(u.T)
            dfnews.append(dfnew)
        hue = "fuzzy"

    barrierfree = False
    if type(markers) == bool:

        if markers is True:
            barrierfree = True
            markers = marker_list
        else:
            markers = []
    if len(category) != 0:
        lut = {}
        for i, cat in enumerate(category):
            if df[cat].dtype == float:
                continue
            _clut, _mlut = _create_color_markerlut(df, cat, palette[1], markers)
            lut[cat] = {"colorlut": _clut, "markerlut": _mlut}

    _dfnews = {}

    if method == "fuzzy" or method == "hdbscan":
        for dfnew, K, fl in zip(dfnews, n_clusters, fuzzylabels):
            if len(category) == 0:
                fig, ax = plt.subplots(ncols=2, figsize=[8, 4])
                ax = [ax]
            else:
                fig, ax = plt.subplots(
                    ncols=2 + len(category), figsize=[8 + 4 * len(category), 4]
                )

            if title != "":
                fig.suptitle(title)

            if type(K) == str:
                _K = K
                K, eps = _K.split(", ")
                K = int(K)
                _cmap = plt.get_cmap(palette[0], K)
            else:
                _cmap = plt.get_cmap(palette[0], K)
            colors = []
            color_entropy = []
            for c in fl:
                tmp = np.zeros([3])
                for i in range(K):
                    # print(_cmap(i))
                    # print(c[i])
                    tmp += np.array(_cmap(i))[:3] * c[i]
                tmp = np.where(tmp > 1, 1, tmp)
                colors.append(tmp)
                color_entropy.append(np.sum(tmp * np.log2(tmp + 0.000001)))

            entropy_srt = np.argsort(color_entropy)
            colors = np.array(colors)[entropy_srt]

            ax[0].scatter(
                dfnew[x].values[entropy_srt],
                dfnew[y].values[entropy_srt],
                color=colors,
                s=size,
            )
            ax[0].set_xlabel(x)
            ax[0].set_ylabel(y)

            if method == "fuzzy":
                _title = "Fuzzy c-means. Cluster num=" + str(K)
            elif method == "hdbscan":
                _title = "HDBSCAN. Cluster num=" + _K
            ax[0].set_title(_title, alpha=0.5)
            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="lavender",
                    label=method + str(i),
                    markerfacecolor=_cmap(i),
                    markersize=10,
                )
                for i in range(K)
            ]

            ax[0].legend(handles=legend_elements, loc="best")
            for i in range(K):
                dfnew[method + str(i)] = fl[:, i]

            pie_scatter(
                dfnew,
                x=x,
                y=y,
                category=[method + str(i) for i in range(K)],
                piesize_scale=piesize_scale,
                ax=ax[1],
                label="",
                bbox_to_anchor="best",
                title="Probability is represented by pie charts",
            )

            if len(category) != 0:
                for i, cat in enumerate(category):
                    dfnew[cat] = df[cat]
                    if dfnew[cat].dtype == float:
                        sc = ax[i + 2].scatter(dfnew[x], dfnew[y], c=dfnew[cat], s=size)
                        plt.colorbar(
                            sc,
                            ax=ax[i + 2],
                            label=cat,
                            shrink=0.3,
                            aspect=5,
                            orientation="vertical",
                        )
                    elif barrierfree is True:
                        for key in lut[cat]["colorlut"].keys():
                            _dfnew = dfnew.loc[dfnew[cat] == key]

                            ax[i + 2].scatter(
                                _dfnew[x],
                                _dfnew[y],
                                color=lut[cat]["colorlut"][key],
                                marker=lut[cat]["markerlut"][key],
                                label=key,
                            )
                        ax[i + 2].legend(title=cat)
                    else:
                        for key in lut[cat]["colorlut"].keys():
                            _dfnew = dfnew.loc[dfnew[cat] == key]
                            ax[i + 2].scatter(
                                _dfnew[x],
                                _dfnew[y],
                                color=lut[cat]["colorlut"][key],
                                label=key,
                                s=size,
                            )
                        ax[i + 2].legend(title=cat)

            _dfnews[K] = dfnew
    else:

        for dfnew, K in zip(dfnews, n_clusters):
            if len(category) == 0:
                # axnum = 1
                fig, ax = plt.subplots(ncols=1, figsize=[4, 4])
                ax = [ax]
            else:
                fig, ax = plt.subplots(
                    ncols=1 + len(category), figsize=[4 + 4 * len(category), 4]
                )
            if title != "":
                fig.suptitle(title)

            if barrierfree is True:
                _clut, _mlut = _create_color_markerlut(dfnew, hue, palette[0], markers)

                for key in _clut.keys():
                    _dfnew = dfnew.loc[dfnew[hue] == key]
                    ax[0].scatter(
                        _dfnew[x],
                        _dfnew[y],
                        color=_clut[key],
                        marker=_mlut[key],
                        label=key,
                    )

            else:

                _clut, _mlut = _create_color_markerlut(dfnew, hue, palette[0], markers)

                for key in _clut.keys():
                    _dfnew = dfnew.loc[dfnew[hue] == key]
                    ax[0].scatter(
                        _dfnew[x], _dfnew[y], color=_clut[key], label=key, s=size
                    )

            ax[0].legend(title=hue)
            ax[0].set_title(method + " Cluster number=" + str(K))
            ax[0].set_xlabel(x)
            ax[0].set_ylabel(y)
            if len(category) != 0:
                for i, cat in enumerate(category):
                    dfnew[cat] = df[cat]

                    if dfnew[cat].dtype == float:
                        sc = ax[i + 1].scatter(
                            dfnew[x], dfnew[y], c=dfnew[cat], label=key, s=size
                        )

                        plt.colorbar(
                            sc,
                            ax=ax[i + 1],
                            label=cat,
                            shrink=0.3,
                            aspect=5,
                            orientation="vertical",
                        )
                    elif barrierfree is True:

                        for key in lut[cat]["colorlut"].keys():
                            _dfnew = dfnew.loc[dfnew[cat] == key]
                            ax[i + 1].scatter(
                                _dfnew[x],
                                _dfnew[y],
                                color=lut[cat]["colorlut"][key],
                                marker=lut[cat]["markerlut"][key],
                                label=key,
                            )
                        ax[i + 1].legend(title=cat)

                    else:
                        for key in lut[cat]["colorlut"].keys():
                            _dfnew = dfnew.loc[dfnew[cat] == key]
                            ax[i + 1].scatter(
                                _dfnew[x],
                                _dfnew[y],
                                color=lut[cat]["colorlut"][key],
                                label=key,
                                s=size,
                            )
                        ax[i + 1].legend(title=cat)

            _dfnews[K] = dfnew
    _save(save, method + "_scatter")
    return {"data": _dfnews, "axes": ax}


def _clusterplot(
    df: pd.DataFrame,
    variables: List = [],
    category: Union[List[str], str] = "",
    method: str = "kmeans",
    n_clusters: Union[str, int] = 3,
    x: str = "",
    y: str = "",
    size: float = 10,
    reduce_dimension: str = "umap",
    testrange: list = [1, 20],
    topn_cluster_num: int = 2,
    show: bool = False,
    min_dist: float = 0.25,
    n_neighbors: int = 15,
    eps: Union[List[float], float] = 0.5,
    pcacomponent: Optional[int] = None,
    ztranform: bool = True,
    palette: list = ["Spectral", "tab20b"],
    save: str = "",
    title: str = "",
    markers: bool = False,
    ax: Optional[plt.Axes] = None,
    piesize_scale: float = 0.02,
    min_cluster_size: int = 10,
    sp_kw={},
    **kwargs,
) -> Dict:
    """
    Clustering data and draw them as a scatter plot optionally with dimensionality reduction.

    Parameters
    ----------
    df : pandas DataFrame
    x, y: str, optional
        The column names to be the x and y axes of scatter plots. If reduce_dimension=True, these options will be
        ignored.

    variables: list, optional
        The names of variables to calculate clusters..

    category: str, optional
        the column name of a known sample category (if exists).
    method: str
        Method name for clustering.
        "kmeans"
        "hierarchical",
        "dbscan": Density-Based Clustering Algorithms
        "fuzzy" : fuzzy c-mean clustering using scikit-fuzzy
    n_clusters: int or str, optional (default: 3)
        The number of clusters to be created. If "auto" is provided, it will estimate optimal
        cluster numbers with "Sum of squared distances" for k-mean clustering and silhouette method for others.
    eps: int or list[int]
        DBSCAN's hyper parameter. It will affect the total number of clusters.
    reduce_dimension: str, optional (default: "umap")
        Dimensionality reduction method. if "" is passed, no reduction methods are applied.
        In this case, data must have only two dimentions or x and y options must be specified.

    markers: bool, optional (default: False)
        Whether to use different markers for each cluster/category (for a colorblind-friendly plot).
    show: bool, optional (default: False)
        Whether to show figures
    size: float, optional (default: 10)
        The size of points in the scatter plot.

    testrange: list, optional (default: [1,20])
        The range of cluster numbers to be tested when n_clusters="auto".
    topn_cluster_num: int, optional (default: 2)
        Top n optimal cluster numbers to be plotted when n_clusters="auto".


    min_dist: float, optional (default: 0.25)
        A UMAP parameter
    n_neighbors: int, optinal (default: 15)
        A UMAP parameter.
    eps: Union[List[float], float], optional (default: 0.5)
        A DBSCAN parameter.
    pcacomponent: Optional[int]=None,
        The number of PCA component. PCA result will be used by UMAP and hierarchical clustering.
    ztranform: bool, optinal (default: True)
        Whether to convert data into z scores.
    palette: list, optional (default: ["Spectral","cubehelix"])

    save: str="",
    piesize_scale: float=0.02
    Returns
    -------
    Raises
    ------
    Notes
    -----
    References
    ----------
    See Also
    --------
    Examples
    --------
    """

    original_index = df.index

    X, category = _separate_data(df, variables=variables, category=category)

    if ztranform is True:
        X = zscore(X, axis=0)

    if pcacomponent is None:

        if 20 < X.shape[1]:
            pcacomponent = 20
        elif 10 < X.shape[1]:
            pcacomponent = 10
        else:
            pcacomponent = 2
    pca = PCA(n_components=pcacomponent, random_state=1)
    xpca = pca.fit_transform(X)

    if reduce_dimension == "umap":
        import umap

        u = umap.UMAP(random_state=42, min_dist=min_dist, n_neighbors=n_neighbors)
        X = u.fit_transform(xpca)
        x = "UMAP1"
        y = "UMAP2"
    elif reduce_dimension == "tsne":
        tsne = _get_embedding("tsne")
        # u=umap.UMAP(random_state=42, min_dist=min_dist,n_neighbors=n_neighbors)
        X = tsne.fit_transform(xpca)
        x = "TSNE1"
        y = "TSNE2"
    if n_clusters == "auto" and method == "kmeans":
        Sum_of_squared_distances = []
        K = list(range(*testrange))
        for k in K:
            km = KMeans(n_clusters=k, n_init=10)
            km = km.fit(X)
            Sum_of_squared_distances.append(km.inertia_)
        normy = np.array(Sum_of_squared_distances) / np.amax(Sum_of_squared_distances)
        normy = 1 - normy
        normx = np.linspace(0, 1, len(K))
        perp = _calc_curveture(normx, normy)
        srtindex = np.argsort(perp)[::-1]
        plt.subplots()
        plt.plot(K, Sum_of_squared_distances, "-", label="Sum of squared distances")
        plt.plot(K, perp * np.amax(Sum_of_squared_distances), label="curveture")
        for i in range(topn_cluster_num):
            plt.plot(
                [K[srtindex[i]], K[srtindex[i]]],
                [0, np.amax(Sum_of_squared_distances)],
                "--",
                color="r",
            )
            plt.text(
                K[srtindex[i]],
                np.amax(Sum_of_squared_distances) * 0.95,
                "N=" + str(K[srtindex[i]]),
            )
        plt.xticks(K)
        plt.xlabel("Cluster number")
        plt.ylabel("Sum of squared distances")
        plt.title("Elbow method for optimal cluster number")
        plt.legend()
        # print("Top two optimal cluster No are: {}, {}".format(K[srtindex[0]],K[srtindex[1]]))
        # n_clusters=[K[srtindex[0]],K[srtindex[1]]]
        n_clusters = [K[i] for i in srtindex[:topn_cluster_num]]
        print("Top two optimal cluster No are:", n_clusters)
        _save(save, method)
    elif n_clusters == "auto" and method == "fuzzy":
        try:
            import skfuzzy as fuzz
        except ImportError:
            from pip._internal import main as pip

            pip(["install", "--user", "scikit-fuzzy"])
            import skfuzzy as fuzz
        fpcs = []
        K = list(range(*testrange))
        _X = X.T
        for nc in K:

            cntr, u, u0, d, jm, p, fpc = fuzz.cmeans(
                _X, nc, 2, error=0.005, maxiter=1000, init=None
            )

            fpcs.append(fpc)

        srtindex = np.argsort(fpcs)[::-1]
        plt.subplots()
        plt.plot(K, fpcs, "-")

        for i in range(topn_cluster_num):
            plt.plot(
                [K[srtindex[i]], K[srtindex[i]]], [0, np.amax(fpcs)], "--", color="r"
            )
            plt.text(K[srtindex[i]], np.amax(fpcs) * 0.95, "N=" + str(K[srtindex[i]]))
        plt.xticks(K)
        plt.xlabel("Cluster number")
        plt.ylabel("Fuzzy partition coefficient")
        n_clusters = [K[i] for i in srtindex[:topn_cluster_num]]
        print("Top two optimal cluster No are:", n_clusters)

        _save(save, method)
    elif n_clusters == "auto" and method == "hierarchical":
        import scipy.spatial.distance as ssd

        labels = df.index
        D = ssd.squareform(ssd.pdist(xpca))
        Y = sch.linkage(D, method="ward")
        Z = sch.dendrogram(Y, labels=labels, no_plot=True)

        K = list(range(*testrange))
        newK = []
        scores = []
        for k in K:
            t = _dendrogram_threshold(Z, k)
            Z2 = sch.dendrogram(Y, labels=labels, color_threshold=t, no_plot=True)
            clusters = _get_cluster_classes(Z2, label="ivl")
            _k = len(clusters)
            if _k not in newK:
                newK.append(_k)
                sample2cluster = {}
                i = 1
                for k, v in clusters.items():
                    for sample in v:
                        sample2cluster[sample] = "C" + str(i)
                    i += 1
                scores.append(
                    silhouette_score(
                        X,
                        [sample2cluster[sample] for sample in labels],
                        metric="euclidean",
                    )
                    / _k
                )

        scores = np.array(scores)
        srtindex = np.argsort(scores)[::-1]
        plt.subplots()
        plt.plot(newK, scores, "-")
        for i in range(topn_cluster_num):
            plt.plot(
                [newK[srtindex[i]], newK[srtindex[i]]],
                [0, np.amax(scores)],
                "--",
                color="r",
            )
            plt.text(
                newK[srtindex[i]], np.amax(scores) * 0.95, "N=" + str(newK[srtindex[i]])
            )
        plt.xticks(newK)
        plt.xlabel("Cluster number")
        plt.ylabel("Silhouette scores")
        plt.title("Optimal cluster number searches by silhouette method")

        n_clusters = [newK[i] for i in srtindex[:topn_cluster_num]]
        print("Top two optimal cluster No are:", n_clusters)
        _save(save, method)
    elif n_clusters == "auto" and method == "dbscan":

        from sklearn.neighbors import NearestNeighbors

        neigh = NearestNeighbors(n_neighbors=2)
        nbrs = neigh.fit(X)
        distances, indices = nbrs.kneighbors(X)
        distances = np.sort(distances[:, 1], axis=0)

        K = np.linspace(np.amin(distances), np.amax(distances), 20)
        newK = []
        scores = []
        _K = []
        for k in K:
            db = DBSCAN(eps=k, min_samples=5, n_jobs=-1)
            dbX = db.fit(X)
            labels = np.unique(dbX.labels_[dbX.labels_ >= 0])

            if len(labels) < 2:
                continue
            _k = len(labels)
            if _k not in newK:
                newK.append(_k)
                _K.append(k)
                scores.append(
                    silhouette_score(
                        X[dbX.labels_ >= 0],
                        dbX.labels_[dbX.labels_ >= 0],
                        metric="euclidean",
                    )
                    / _k
                )

        scores = np.array(scores)

        _ksort = np.argsort(newK)
        _K = np.array(_K)[_ksort]
        newK = np.array(newK)[_ksort]
        scores = np.array(scores)[_ksort]
        srtindex = np.argsort(scores)[::-1]
        plt.subplots()
        plt.plot(newK, scores, "-")

        for i in range(topn_cluster_num):
            plt.plot(
                [_K[srtindex[i]], _K[srtindex[i]]],
                [0, np.amax(scores)],
                "--",
                color="r",
            )
            plt.text(
                _K[srtindex[i]], np.amax(scores) * 0.95, "N=" + str(newK[srtindex[i]])
            )
        plt.xticks(newK)
        plt.xlabel("eps")
        plt.ylabel("Silhouette scores")
        plt.title("Optimal cluster number searches by silhouette method")

        _n_clusters = [newK[i] for i in range(topn_cluster_num)]
        print("Top two optimal cluster No are:", _n_clusters)
        eps = [_K[i] for i in srtindex[:topn_cluster_num]]
        _save(save, method)

    elif n_clusters == "auto" and method == "hdbscan":
        try:
            import hdbscan
        except ImportError:
            from pip._internal import main as pip

            pip(["install", "--user", "hdbscan"])
            import hdbscan

        from sklearn.neighbors import NearestNeighbors

        neigh = NearestNeighbors(n_neighbors=2)
        nbrs = neigh.fit(X)
        distances, indices = nbrs.kneighbors(X)
        distances = np.sort(distances[:, 1], axis=0)

        # K=np.linspace(0.01,1,10)
        K = np.arange(2, 20, 1)
        print(K)
        newK = []
        scores = []
        _K = []
        for k in K:
            db = hdbscan.HDBSCAN(
                min_cluster_size=k,
                # cluster_selection_epsilon=k,
                algorithm="best",
                alpha=1.0,
                leaf_size=40,
                metric="euclidean",
                min_samples=None,
                p=None,
                core_dist_n_jobs=-1,
            )
            dbX = db.fit(X)
            labels = np.unique(dbX.labels_[dbX.labels_ >= 0])

            if len(labels) < 2:
                continue
            _k = len(labels)
            if _k not in newK:
                newK.append(_k)
                _K.append(k)
                scores.append(
                    silhouette_score(
                        X[dbX.labels_ >= 0],
                        dbX.labels_[dbX.labels_ >= 0],
                        metric="euclidean",
                    )
                    / _k
                )

        scores = np.array(scores)

        _ksort = np.argsort(newK)
        _K = np.array(_K)[_ksort]
        newK = np.array(newK)[_ksort]
        scores = np.array(scores)[_ksort]
        srtindex = np.argsort(scores)[::-1]
        plt.subplots()
        plt.plot(newK, scores, "-")

        for i in range(topn_cluster_num):
            plt.plot(
                [_K[srtindex[i]], _K[srtindex[i]]],
                [0, np.amax(scores)],
                "--",
                color="r",
            )
            plt.text(
                _K[srtindex[i]], np.amax(scores) * 0.95, "N=" + str(newK[srtindex[i]])
            )
        plt.xticks(_K)
        plt.xlabel("min_cluster_size")
        plt.ylabel("Silhouette scores")
        plt.title("Optimal cluster number searches by silhouette method")

        _n_clusters = [newK[i] for i in range(topn_cluster_num)]
        print("Top two optimal cluster No are:", _n_clusters)
        min_cluster_size = [_K[i] for i in srtindex[:topn_cluster_num]]
        _save(save, method)
    else:
        n_clusters = [n_clusters]
    if method == "kmeans":
        dfnews = []

        for nc in n_clusters:
            kmean = KMeans(n_clusters=nc, random_state=0, n_init=10)
            kmX = kmean.fit(X)
            labels = np.unique(kmX.labels_)

            dfnew = pd.DataFrame(
                data=np.array([X[:, 0], X[:, 1]]).T,
                columns=[x, y],
                index=original_index,
            )
            dfnew["kmeans"] = kmX.labels_
            dfnews.append(dfnew)
        hue = "kmeans"

    elif method == "hierarchical":
        import scipy.spatial.distance as ssd

        labels = df.index
        D = ssd.squareform(ssd.pdist(xpca))
        Y = sch.linkage(D, method="ward")
        Z = sch.dendrogram(Y, labels=labels, no_plot=True)

        dfnews = []
        for nc in n_clusters:
            t = _dendrogram_threshold(Z, nc)
            Z2 = sch.dendrogram(Y, labels=labels, color_threshold=t, no_plot=True)
            clusters = _get_cluster_classes(Z2, label="ivl")
            sample2cluster = {}
            i = 1
            for k, v in clusters.items():
                for sample in v:
                    sample2cluster[sample] = "C" + str(i)
                i += 1

            dfnew = pd.DataFrame(
                data=np.array([X[:, 0], X[:, 1]]).T,
                columns=[x, y],
                index=original_index,
            )
            dfnew["hierarchical"] = [sample2cluster[sample] for sample in labels]
            dfnews.append(dfnew)
        hue = "hierarchical"
    elif method == "dbscan":
        dfnews = []

        if type(eps) == float:
            eps = [eps]
        n_clusters = []
        for e in eps:
            db = DBSCAN(eps=e, min_samples=5, n_jobs=-1)
            dbX = db.fit(X)
            labels = np.unique(dbX.labels_)

            dfnew = pd.DataFrame(
                data=np.array([X[:, 0], X[:, 1]]).T,
                columns=[x, y],
                index=original_index,
            )
            dfnew["dbscan"] = dbX.labels_
            dfnews.append(dfnew)
            tmp = 0
            for c in set(dbX.labels_):
                if c >= 0:
                    tmp += 1
            n_clusters.append(str(tmp) + ", eps=" + str(np.round(e, 2)))

        hue = "dbscan"
    elif method == "hdbscan":
        dfnews = []

        try:
            import hdbscan
        except ImportError:
            from pip._internal import main as pip

            pip(["install", "--user", "hdbscan"])
            import hdbscan

        if type(min_cluster_size) == int:
            min_cluster_size = [min_cluster_size]
        n_clusters = []
        fuzzylabels = []
        for e in min_cluster_size:
            db = hdbscan.HDBSCAN(
                min_cluster_size=e,
                prediction_data=True,
                algorithm="best",
                alpha=1.0,
                approx_min_span_tree=True,
                gen_min_span_tree=True,
                leaf_size=40,
                metric="euclidean",
                min_samples=None,
                p=None,
            )
            dbX = db.fit(X)
            labels = np.unique(dbX.labels_)

            dfnew = pd.DataFrame(
                data=np.array([X[:, 0], X[:, 1]]).T,
                columns=[x, y],
                index=original_index,
            )
            dfnew["dbscan"] = dbX.labels_
            fuzzylabels.append(hdbscan.all_points_membership_vectors(dbX))
            dfnews.append(dfnew)
            tmp = 0
            for c in set(dbX.labels_):
                if c >= 0:
                    tmp += 1
            n_clusters.append(str(tmp) + ", eps=" + str(np.round(e, 2)))

        hue = "hdbscan"
    elif method == "fuzzy":
        try:
            import skfuzzy as fuzz
        except ImportError:
            from pip._internal import main as pip

            pip(["install", "--user", "scikit-fuzzy"])
            import skfuzzy as fuzz

        dfnews = []
        fuzzylabels = []

        _X = X.T
        for nc in n_clusters:

            cntr, u, u0, d, jm, p, fpc = fuzz.cmeans(
                _X, nc, 2, error=0.005, maxiter=1000, init=None
            )

            dfnew = pd.DataFrame(
                data=np.array([X[:, 0], X[:, 1]]).T,
                columns=[x, y],
                index=original_index,
            )
            fuzzylabels.append(u.T)
            dfnews.append(dfnew)
        hue = "fuzzy"

    barrierfree = False
    if type(markers) == bool:

        if markers is True:
            barrierfree = True
            markers = marker_list
        else:
            markers = []
    if len(category) != 0:
        lut = {}
        for i, cat in enumerate(category):
            if df[cat].dtype == float:
                continue
            _clut, _mlut = _create_color_markerlut(df, cat, palette[1], markers)
            lut[cat] = {"colorlut": _clut, "markerlut": _mlut}

    _dfnews = {}

    if method == "fuzzy" or method == "hdbscan":
        for dfnew, K, fl in zip(dfnews, n_clusters, fuzzylabels):
            if len(category) == 0:
                fig, ax = plt.subplots(ncols=2, figsize=[8, 4])
                ax = [ax]
            else:
                fig, ax = plt.subplots(
                    ncols=2 + len(category), figsize=[8 + 4 * len(category), 4]
                )

            if title != "":
                fig.suptitle(title)

            if type(K) == str:
                _K = K
                K, eps = _K.split(", ")
                K = int(K)
                _cmap = plt.get_cmap(palette[0], K)
            else:
                _cmap = plt.get_cmap(palette[0], K)
            colors = []
            color_entropy = []
            for c in fl:
                tmp = np.zeros([3])
                for i in range(K):
                    # print(_cmap(i))
                    # print(c[i])
                    tmp += np.array(_cmap(i))[:3] * c[i]
                tmp = np.where(tmp > 1, 1, tmp)
                colors.append(tmp)
                color_entropy.append(np.sum(tmp * np.log2(tmp + 0.000001)))

            entropy_srt = np.argsort(color_entropy)
            colors = np.array(colors)[entropy_srt]

            # ax[0].scatter(dfnew[x].values[entropy_srt], dfnew[y].values[entropy_srt], color=colors, s=size)
            # ax[0].set_xlabel(x)
            # ax[0].set_ylabel(y)

            _tmp = dfnew.iloc[entropy_srt]

            res = scatterplot(
                df=_tmp, ax=ax[0], x=x, y=y, c=colors, axlabel="each", fig=fig, **sp_kw
            )
            fig = res["fig"]
            ax[0] = res["axes"][0]
            # sns.scatterplot(data=dfnew,x=x,y=y,hue=hue, ax=ax[0], palette=palette[0],**kwargs)
            if method == "fuzzy":
                _title = "Fuzzy c-means. Cluster num=" + str(K)
            elif method == "hdbscan":
                _title = "HDBSCAN. Cluster num=" + _K
            ax[0].set_title(_title, alpha=0.5)
            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="lavender",
                    label=method + str(i),
                    markerfacecolor=_cmap(i),
                    markersize=10,
                )
                for i in range(K)
            ]

            ax[0].legend(handles=legend_elements, loc="best")
            for i in range(K):
                dfnew[method + str(i)] = fl[:, i]

            pie_scatter(
                dfnew,
                x=x,
                y=y,
                category=[method + str(i) for i in range(K)],
                piesize_scale=piesize_scale,
                ax=ax[1],
                label="",
                bbox_to_anchor="best",
                title="Probability is represented by pie charts",
            )

            if len(category) != 0:
                for i, cat in enumerate(category):
                    dfnew[cat] = df[cat]
                    # sns.scatterplot(data=dfnew,x=x,y=y,hue=cat, ax=ax[i+2], palette=palette[1], s=size,**kwargs)
                    if dfnew[cat].dtype == float:
                        res = scatterplot(
                            df=dfnew,
                            ax=ax[i + 2],
                            fig=fig,
                            x=x,
                            y=y,
                            colors=cat,
                            axlabel="non",
                            show_legend=False,
                            **sp_kw,
                        )
                        fig = res["fig"]
                        # sc=ax[i+2].scatter(dfnew[x], dfnew[y], c=dfnew[cat], s=size)
                        # plt.colorbar(sc,ax=ax[i+2], label=cat, shrink=0.3,aspect=5,orientation="vertical")
                    elif barrierfree is True:
                        res = scatterplot(
                            df=dfnew,
                            ax=ax[i + 2],
                            fig=fig,
                            x=x,
                            y=y,
                            category=cat,
                            palette=palette[1],
                            axlabel="non",
                            markers=True,
                            show_legend=False,
                            **sp_kw,
                        )
                        fig = res["fig"]
                        # for key in lut[cat]["colorlut"].keys():
                        #     _dfnew=dfnew.loc[dfnew[cat]==key]

                        #     ax[i+2].scatter(_dfnew[x], _dfnew[y], color=lut[cat]["colorlut"][key], marker=lut[cat]["markerlut"][key], label=key)
                        ax[i + 2].legend(title=cat)
                    else:
                        res = scatterplot(
                            df=dfnew,
                            ax=ax[i + 2],
                            fig=fig,
                            x=x,
                            y=y,
                            category=cat,
                            palette=palette[1],
                            axlabel="non",
                            show_legend=False,
                            **sp_kw,
                        )
                        fig = res["fig"]
                        # for key in lut[cat]["colorlut"].keys():
                        #     _dfnew=dfnew.loc[dfnew[cat]==key]
                        #     ax[i+2].scatter(_dfnew[x], _dfnew[y], color=lut[cat]["colorlut"][key], label=key, s=size)
                        ax[i + 2].legend(title=cat)

            _dfnews[K] = dfnew
    else:

        for dfnew, K in zip(dfnews, n_clusters):
            if len(category) == 0:
                # axnum = 1
                fig, ax = plt.subplots(ncols=1, figsize=[4, 4])
                ax = [ax]
            else:
                fig, ax = plt.subplots(
                    ncols=1 + len(category), figsize=[4 + 4 * len(category), 4]
                )
            if title != "":
                fig.suptitle(title)

            if barrierfree is True:
                _clut, _mlut = _create_color_markerlut(dfnew, hue, palette[0], markers)

                for key in _clut.keys():
                    _dfnew = dfnew.loc[dfnew[hue] == key]
                    ax[0].scatter(
                        _dfnew[x],
                        _dfnew[y],
                        color=_clut[key],
                        marker=_mlut[key],
                        label=key,
                    )

            else:

                _clut, _mlut = _create_color_markerlut(dfnew, hue, palette[0], markers)

                for key in _clut.keys():
                    _dfnew = dfnew.loc[dfnew[hue] == key]
                    ax[0].scatter(
                        _dfnew[x], _dfnew[y], color=_clut[key], label=key, s=size
                    )

            ax[0].legend(title=hue)
            ax[0].set_title(method + " Cluster number=" + str(K))
            ax[0].set_xlabel(x)
            ax[0].set_ylabel(y)
            if len(category) != 0:
                for i, cat in enumerate(category):
                    dfnew[cat] = df[cat]

                    if dfnew[cat].dtype == float:
                        sc = ax[i + 1].scatter(
                            dfnew[x], dfnew[y], c=dfnew[cat], label=key, s=size
                        )

                        plt.colorbar(
                            sc,
                            ax=ax[i + 1],
                            label=cat,
                            shrink=0.3,
                            aspect=5,
                            orientation="vertical",
                        )
                    elif barrierfree is True:

                        for key in lut[cat]["colorlut"].keys():
                            _dfnew = dfnew.loc[dfnew[cat] == key]
                            ax[i + 1].scatter(
                                _dfnew[x],
                                _dfnew[y],
                                color=lut[cat]["colorlut"][key],
                                marker=lut[cat]["markerlut"][key],
                                label=key,
                            )
                        ax[i + 1].legend(title=cat)

                    else:
                        for key in lut[cat]["colorlut"].keys():
                            _dfnew = dfnew.loc[dfnew[cat] == key]
                            ax[i + 1].scatter(
                                _dfnew[x],
                                _dfnew[y],
                                color=lut[cat]["colorlut"][key],
                                label=key,
                                s=size,
                            )
                        ax[i + 1].legend(title=cat)

            _dfnews[K] = dfnew
    _save(save, method + "_scatter")
    return {"data": _dfnews, "axes": ax}


def pie_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    category: list,
    pie_palette: str = "tab20c",
    label: Union[List, str] = "all",
    topn: int = 10,
    ax: Optional[plt.Axes] = None,
    piesizes: Union[List, str] = "",
    save: str = "",
    show: bool = False,
    edge_color: str = "gray",
    min_piesize: float = 0.3,
    figsize: list = [6, 6],
    xunit: str = "",
    yunit: str = "",
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    logscalex: bool = False,
    logscaley: bool = False,
    bbox_to_anchor: Union[List, str] = [0.95, 1],
    piesize_scale: float = 0.01,
) -> dict:
    """
    Drawing a scatter plot of which points are represented by pie charts.

    Parameters
    ----------
    df : pandas DataFrame
        A wide form dataframe. Index names are used to label points
        e.g.)
                    gas    coal    nuclear    population    GDP
            USA      20      20          5            20     50
            China    30      40          5            40     50
            India     5      10          1            40     10
            Japan     5       5          1            10     10

    x,y : str
        the names of columns to be x and y axes of the scatter plot.
        e.g.)
            x="population", y="GDP"

    category: str or list
        the names of categorical values to display as pie charts
        e.g.)
            category=["gas", "coal", "nuclear"]
    pie_palette : str
        A colormap name
    xlabel: str, optional
        x axis label
    ylabel: str, optional
        y axis label
    piesize: float, optional (default: 0.01)
        pie chart size.
    label: str, optional (default: "all")
        "all": all
        "topn_of_sum": top n samples are labeled
        "": no labels
    logscalex, logscaley: bool, optional (default: False)
        Whether to scale x an y axes with logarithm
    ax: Optional[plt.Axes] optional, (default: None)
        pyplot ax to add this scatter plot
    sizes: Union[List, str], optional (default: "")
        pie chart sizes.
            "sum_of_each": automatically set pie chart sizes to be proportional to the sum of all categories.
            list: the list of pie chart sizes
    edge_color: str="gray",
        The pie chart edge color
    min_piesize: float, optional (default: 0.3)
        Minimal pie chart size. This option is effective when the option sizes="sum_of_each".
    Returns
    -------
    dict

    Raises
    ------
    Notes
    -----
    References
    ----------
    See Also
    --------
    Examples
    --------
    """
    if type(pie_palette) == str:
        colors = {}
        unique_labels = category

        cmap = plt.get_cmap(pie_palette)
        labelnum = len(unique_labels)
        for i, ul in enumerate(unique_labels):
            colors[ul] = cmap(i / labelnum)
    elif type(pie_palette) == dict:
        colors = pie_palette
        unique_labels = colors.keys()
    else:
        raise Exception("Unknown pie_palette type.")
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(right=0.80)
    X = df[x]
    Y = df[y]
    yscale = ""
    xscale = ""
    if logscaley is True:
        Y = np.log10(Y + 1)
        yscale = " (scaled by log10)"
    if logscalex is True:
        X = np.log10(X + 1)
        xscale = " (scaled by log10)"
    Frac = df[category]

    index = df.index
    piesize_scale = np.amax([np.amax(X), np.amax(Y)]) * piesize_scale

    if piesizes == "sum_of_each":
        sums = Frac.sum(axis=1)
        sumsrt = np.argsort(sums)[::-1]
        sumsrt = set(sumsrt[:topn])
        sums = sums / np.amax(sums)
        sums = piesize_scale * (sums + min_piesize)
    _colors = [colors[f] for f in unique_labels]
    for i, (_x, _y, _ind) in enumerate(zip(X, Y, index)):
        _frac = Frac.loc[_ind].values
        _frac = 2 * np.pi * np.array(_frac) / np.sum(_frac)

        angle = 0
        # print(sums.loc[_ind])
        for fr, co in zip(_frac, _colors):
            if type(piesizes) == str:
                if piesizes == "sum_of_each":
                    _baumkuchen_xy(
                        ax,
                        _x,
                        _y,
                        angle,
                        fr,
                        0,
                        sums.loc[_ind],
                        20,
                        co,
                        edgecolor=edge_color,
                    )
                elif piesizes == "":
                    _baumkuchen_xy(
                        ax,
                        _x,
                        _y,
                        angle,
                        fr,
                        0,
                        piesize_scale,
                        20,
                        co,
                        edgecolor=edge_color,
                    )
                else:
                    pass
            elif type(piesizes) == list and len(piesizes) != 0:
                _baumkuchen_xy(
                    ax,
                    _x,
                    _y,
                    angle,
                    fr,
                    0,
                    piesize_scale * piesizes[i],
                    20,
                    co,
                    edgecolor=edge_color,
                )
            else:
                _baumkuchen_xy(
                    ax,
                    _x,
                    _y,
                    angle,
                    fr,
                    0,
                    piesize_scale,
                    20,
                    co,
                    edgecolor=edge_color,
                )
            angle += fr

        if type(label) == str:
            if label == "all":
                ax.text(_x, _y, _ind)
            elif label == "topn_of_sum":
                if i in sumsrt:
                    ax.text(_x, _y, _ind)

            elif label == "":
                pass
        elif type(label) == list:
            if _ind in label:
                ax.text(_x, _y, _ind)

    if xlabel != "":
        x = xlabel
    if ylabel != "":
        y = ylabel
    plt.xlabel(x + xscale)
    plt.ylabel(y + yscale)
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="lavender",
            label=ul,
            markerfacecolor=colors[ul],
            markersize=10,
        )
        for ul in unique_labels
    ]
    if type(bbox_to_anchor) == str:
        ax.legend(handles=legend_elements, loc=bbox_to_anchor)
    else:
        ax.legend(handles=legend_elements, bbox_to_anchor=bbox_to_anchor)
    ax.set_title(title)
    if yunit != "":
        ax.text(0, 1, "({})".format(yunit), transform=ax.transAxes, ha="right")
    if xunit != "":
        ax.text(1, 0, "({})".format(xunit), transform=ax.transAxes, ha="left", va="top")
    _save(save, "pie_scatter")

    if show is True:
        plt.show()
    return {"axes": ax}


def decomplot(
    df: pd.DataFrame,
    variables: List = [],
    category: Union[List, str] = "",
    method: str = "pca",
    component: int = 3,
    arrow_color: str = "yellow",
    arrow_text_color: str = "black",
    show: bool = False,
    explained_variance: bool = True,
    arrow_num: int = 3,
    figsize=[],
    regularization: bool = True,
    pcaparam={"random_state": 0},
    nmfparam={"random_state": 0},
    save: str = "",
    title: str = "",
    markers: bool = False,
    saveparam: dict = {},
    ax: Optional[plt.Axes] = None,
    palette: str = "tab20b",
    size: int = 10,
    show_labels: bool = False,
) -> Dict:

    """
    Decomposing data and drawing a scatter plot and some plots for explained variables.

    Parameters
    ----------
    df : pandas DataFrame

    category: str, optional
        the column name of a known sample category (if exists).

    variables: list, optional
        The names of variables to calculate decomposition.

    method: str, optional (default: "pca")
        Method name for decomposition. Available methods: ["pca", "nmf"]

    component: int, optional (default: 3)
        The component number

    arrow_color: str, optional (default: "yellow")
        The color of eigen vectors

    arrow_text_color: str, optional (default: "black")
        The color of the labels of eigen vectors

    explained_variance: bool, optional (default: True)
        Wheter to plot explained_variance to interpret the PCA/NMF results.

    arrow_num: int, optional (default: 3)
        The number of eigen vectors to be displayed

    figsize: list, optional
        The figure size

    regularization: bool, optional (default: True)
        Whether to regularize the data (z score for PCA and zero to one scaling for NMF)

    pcaparam: dict, optional (default: {"random_state":0})
        The hyperparameter for PCA.

    nmfparam: dict, optional (default: {"random_state":0})
        The hyperparameter for NFM.

    save: str, optional
        if you give a filename to this option, it will save produced figures.

    title: str, optional
        A figure title.

    markers: bool, optional (default: False)
        Whether to make points to be different shapes

    saveparam: dict, optional
        Parameters for saving figures

    ax: Optional[plt.Axes], optional (default: None)
        Axis object to plot

    palette: str, optional (default: "tab20b")
        A matplotlib colormap name

    size: int, optional (default: 10)
        scatter point size

    show: bool
        Whether or not to show the figure.

    show_labels: bool, optional (default: False)
        Whether to show labels on points

    Returns
    -------
        {"data": dfpc_list,"pca": pca, "axes":axes, "axes_explained":ax2}: dict for pca method
        {"data": dfpc_list, "W":W, "H":H,"axes":axes,"axes_explained":axes2}: dict for nmf method


    Raises
    ------
    Notes
    -----
    References
    ----------
    See Also
    --------
    Examples
    --------
    """
    x, category = _separate_data(df, variables=variables, category=category)
    sample_labels = list(df.index)
    barrierfree = False
    if type(markers) == bool:

        if markers is True:
            barrierfree = True
            markers = marker_list
        else:
            markers = []

    original_index = df.index
    if len(variables) != 0:
        features = variables
    else:

        features = sorted(list(set(df.columns) - set(category)))
    dfpc_list = []
    comb = list(combinations(np.arange(component), 2))

    if len(category) != 0:
        lut = {}
        for i, cat in enumerate(category):
            _clut, _mlut = _create_color_markerlut(df, cat, palette, markers)
            lut[cat] = {"colorlut": _clut, "markerlut": _mlut}

    if len(category) != 0:
        figures = {}
        for cat in category:
            if len(comb) == 1:
                fig, axes = plt.subplots()
                axes = [axes]
            else:
                nrows = len(comb) // 2 + int(len(comb) % 2 != 0)
                if len(figsize) == 0:
                    figsize = [8, 3 * nrows]
                ncols = 2
                fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
                plt.subplots_adjust(top=0.9, right=0.8, left=0.1)
                axes = axes.flatten()
                if len(comb) != ncols * nrows:
                    for i in range(ncols * nrows - len(comb)):
                        fig.delaxes(axes[-(i + 1)])
            figures[cat] = {"fig": fig, "axes": axes}
    else:
        figures = {}
        if len(comb) == 1:
            fig, axes = plt.subplots()
            axes = [axes]
        else:
            nrows = len(comb) // 2 + int(len(comb) % 2 != 0)
            if len(figsize) == 0:
                figsize = [8, 3 * nrows]

            fig, axes = plt.subplots(ncols=2, nrows=nrows, figsize=figsize)
            plt.subplots_adjust(top=0.9, right=0.8, left=0.1)
            axes = axes.flatten()
        figures["nocat"] = {"fig": fig, "axes": axes}
    if method == "pca":
        if regularization is True:
            x = zscore(x, axis=0)
        pca = PCA(n_components=component, **pcaparam)
        pccomp = pca.fit_transform(x)
        exp_var_pca = pca.explained_variance_ratio_
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        combnum = 0
        for axi, (i, j) in enumerate(comb):
            xlabel, ylabel = "pc{} ({:.01f}%)".format(
                i + 1, 100 * exp_var_pca[i]
            ), "pc{} ({:.01f}%)".format(j + 1, 100 * exp_var_pca[j])
            dfpc = pd.DataFrame(
                data=np.array([pccomp[:, i], pccomp[:, j]]).T,
                columns=[xlabel, ylabel],
                index=original_index,
            )
            _loadings = np.array([loadings[:, i], loadings[:, j]]).T
            a = np.sum(_loadings**2, axis=1)
            srtindx = np.argsort(a)[::-1][:arrow_num]
            _loadings = _loadings[srtindx]
            _features = np.array(features)[srtindx]

            if len(category) != 0:
                ax = figures[cat]["axes"][axi]
                for cat in category:
                    dfpc[cat] = df[cat]

                    _scatter(
                        dfpc,
                        xlabel,
                        ylabel,
                        cat,
                        ax,
                        lut,
                        barrierfree,
                        size,
                        legend=False,
                    )

                    # sns.scatterplot(data=dfpc, x=xlabel, y=ylabel, hue=cat, ax=figures[cat]["axes"][axi],palette=palette)
                    # sns.scatterplot(data=dfpc, x=xlabel, y=ylabel, hue=cat, ax=figures[cat]["axes"][axi],legend=False,palette=palette)
                    if combnum == 1:
                        ax.legend(
                            bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0
                        )
                    for k, feature in enumerate(_features):
                        ax.arrow(
                            0,
                            0,
                            _loadings[k, 0],
                            _loadings[k, 1],
                            color=arrow_color,
                            width=0.005,
                            head_width=0.1,
                        )
                        ax.text(
                            _loadings[k, 0],
                            _loadings[k, 1],
                            feature,
                            color=arrow_text_color,
                        )
            else:
                ax = figures["nocat"]["axes"][axi]
                sns.scatterplot(x=dfpc[xlabel], y=dfpc[ylabel], ax=ax, palette=palette)
                # cat=None
                # _scatter(dfpc, xlabel,ylabel, cat, figures["nocat"][axi], lut, barrierfree, size)
                for k, feature in enumerate(_features):
                    # ax.plot([0,_loadings[k, 0] ], [0,_loadings[k, 1] ],color=arrow_color)
                    ax.arrow(
                        0,
                        0,
                        _loadings[k, 0],
                        _loadings[k, 1],
                        color=arrow_color,
                        width=0.005,
                        head_width=0.1,
                    )
                    ax.text(
                        _loadings[k, 0],
                        _loadings[k, 1],
                        feature,
                        color=arrow_text_color,
                    )

            if show_labels is True:
                for s, __x, __y in zip(sample_labels, pccomp[:, i], pccomp[:, j]):
                    ax.text(__x, __y, s)

            dfpc_list.append(dfpc)
            combnum += 1

        if len(category) != 0:
            for cat in category:
                figures[cat]["fig"].suptitle(title)
                figures[cat]["fig"].tight_layout(pad=0.5)
                _save(save, cat + "_PCA", fig=figures[cat]["fig"])
        else:
            figures["nocat"]["fig"].suptitle(title)
            figures["nocat"]["fig"].tight_layout(pad=0.5)
            _save(save, "PCA")

        if explained_variance is True:
            fig, ax2 = plt.subplots(nrows=2)
            exp_var_pca = pca.explained_variance_ratio_
            #
            # Cumulative sum of eigenvalues; This will be used to create step plot
            # for visualizing the variance explained by each principal component.
            #
            cum_sum_eigenvalues = np.cumsum(exp_var_pca)
            #
            # Create the visualization plot
            #
            xlabel = ["pc" + str(i + 1) for i in range(0, len(exp_var_pca))]
            ax2[0].bar(
                xlabel,
                exp_var_pca,
                alpha=0.5,
                align="center",
                label="Individual explained variance",
            )
            ax2[0].step(
                range(0, len(cum_sum_eigenvalues)),
                cum_sum_eigenvalues,
                where="mid",
                label="Cumulative explained variance",
            )
            ax2[0].set_ylabel("Explained variance ratio")
            ax2[0].set_xlabel("Principal component index")

            components = pca.components_.T
            imy = np.amin([10, len(_features)])
            components = components[:imy]
            vmax = np.abs(components).max()
            pos = ax2[1].imshow(
                components, cmap="coolwarm", vmax=vmax, vmin=-vmax, aspect="auto"
            )
            ax2[1].set_yticks(np.arange(imy))
            ax2[1].set_yticklabels(_features[:imy])
            ax2[1].set_title(str("Loadings"))
            ax2[1].set_xticks(np.arange(len(xlabel)))
            ax2[1].set_xticklabels(xlabel)
            ax2[1].grid(False)
            fig.colorbar(pos, ax=ax2[1])
            plt.tight_layout(h_pad=1)
            _save(save, "ExplainedVar")

        else:
            ax2 = None
        if show is True:
            plt.show()
        return {"data": dfpc_list, "pca": pca, "axes": figures, "axes_explained": ax2}
    elif method == "nmf":
        nmf = NMF(n_components=component, **nmfparam)
        if regularization is True:
            x = x / np.sum(x, axis=0)[None, :]
        W = nmf.fit_transform(x)
        H = nmf.components_
        combnum = 0
        for axi, (i, j) in enumerate(comb):
            xlabel, ylabel = "p" + str(i + 1), "p" + str(j + 1)
            dfpc = pd.DataFrame(
                data=np.array([W[:, i], W[:, j]]).T,
                columns=[xlabel, ylabel],
                index=original_index,
            )
            if len(category) != 0:
                for cat in category:
                    dfpc[cat] = df[cat]

                    # sns.scatterplot(data=dfpc, x=xlabel, y=ylabel, hue=cat, ax=figures[cat]["axes"][axi],palette=palette)
                    _scatter(
                        dfpc,
                        xlabel,
                        ylabel,
                        cat,
                        figures[cat]["axes"][axi],
                        lut,
                        barrierfree,
                        size,
                        legend=False,
                    )
                    # sns.scatterplot(data=dfpc, x=xlabel, y=ylabel, hue=cat, ax=figures[cat]["axes"][axi],legend=False,palette=palette)
                    if combnum == 1:
                        figures[cat]["axes"][axi].legend(
                            bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0
                        )

            else:
                sns.scatterplot(
                    data=dfpc,
                    x=xlabel,
                    y=ylabel,
                    hue=category,
                    ax=figures["nocat"]["axes"][axi],
                    palette=palette,
                )
                # cat=None
                # _scatter(dfpc, xlabel,ylabel, cat, figures["nocat"][axi], lut, barrierfree, size)
            if show_labels is True:
                for s, __x, __y in zip(sample_labels, W[:, i], W[:, j]):
                    ax.text(__x, __y, s)
            dfpc_list.append(dfpc)
            combnum += 1
        if len(category) != 0:
            for cat in category:
                figures[cat]["fig"].suptitle(title)
                figures[cat]["fig"].tight_layout(pad=0.5)
                _save(save, cat + "_NMF", fig=figures[cat]["fig"])
        else:
            figures["nocat"]["fig"].suptitle(title)
            figures["nocat"]["fig"].tight_layout(pad=0.5)
        _save(save, "NMF")
        if explained_variance is True:
            fig, axes2 = plt.subplots(nrows=component, figsize=[5, 5])
            axes2 = axes2.flatten()
            for i, ax in enumerate(axes2):
                if i == 0:
                    ax.set_title("Coefficients of matrix H")
                ax.bar(np.arange(len(features)), H[i])
                ax.set_ylabel("p" + str(i + 1))
                ax.set_xticks(np.arange(len(features)), labels=[])
            ax.set_xticks(np.arange(len(features)), labels=features, rotation=90)
            fig.tight_layout()

            # dfw={"index":[],"p":[],"val":[]}
            # ps=["p"+str(i+1) for i in range(component)]
            # originalindex=df.index
            # for i in range(W.shape[0]):
            #     for j in range(W.shape[1]):
            #         dfw["index"].append(originalindex[i])
            #         dfw["p"].append(ps[j])
            #         dfw["val"].append(W[i,j])
            # dfw=pd.DataFrame(data=dfw)
            #
            # dfh={"feature":[],"p":[],"val":[]}
            # for i in range(H.shape[0]):
            #     for j in range(H.shape[1]):
            #         dfh["p"].append(ps[i])
            #         dfh["feature"].append(features[j])
            #
            #         dfh["val"].append(H[i,j])
            # dfw=pd.DataFrame(data=dfw)
            # dfh=pd.DataFrame(data=dfh)
            # #dotplot(dfw,row="index",col="p",size_val="val")
            # dotplot(dfh,row="p",col="feature",size_val="val",)
            _save(save, "Coefficients")
        else:
            axes2 = None
        if show is True:
            plt.show()
        return {
            "data": dfpc_list,
            "W": W,
            "H": H,
            "axes": figures,
            "axes_explained": axes2,
        }
    elif method == "lda":
        _ = LatentDirichletAllocation(n_components=component, random_state=0)
        if regularization:
            x = x / np.sum(x, axis=0)[None, :]

    else:
        raise Exception(
            "{} is not in options. Available options are: pca, nmf".format(method)
        )


def _decomplot(
    df: pd.DataFrame,
    variables: List = [],
    category: Union[List, str] = "",
    method: str = "pca",
    component: int = 3,
    arrow_color: str = "yellow",
    arrow_text_color: str = "black",
    show: bool = False,
    explained_variance: bool = True,
    arrow_num: int = 3,
    figsize=[],
    regularization: bool = True,
    pcapram={"random_state": 0},
    nmfparam={"random_state": 0},
    save: str = "",
    title: str = "",
    markers: bool = False,
    saveparam: dict = {},
    ax: Optional[plt.Axes] = None,
    palette: str = "tab20b",
    size: int = 10,
) -> Dict:

    """
    Decomposing data and drawing a scatter plot and some plots for explained variables.

    Parameters
    ----------
    df : pandas DataFrame

    category: str
        the column name of a known sample category (if exists).
    variables: list, optional
        The names of variables to calculate decomposition.
    method: str
        Method name for decomposition. Available methods: ["pca", "nmf"]
    component: int
        The component number

    show : bool
        Whether or not to show the figure.

    Returns
    -------
        dict {"data": dfpc_list,"pca": pca, "axes":axes, "axes_explained":ax2} for pca method
        or {"data": dfpc_list, "W":W, "H":H,"axes":axes,"axes_explained":axes2} for nmf method


    Raises
    ------
    Notes
    -----
    References
    ----------
    See Also
    --------
    Examples
    --------
    """
    x, category = _separate_data(df, variables=variables, category=category)
    # if category !="":
    #     category_val=df[category].values
    #     df=df.drop([category], axis=1)
    #     x = df.values
    #     assert x.dtype==float, f"data must contain only float values except {category} column."
    #
    # else:
    #     x = df.values
    #     assert x.dtype==float, "data must contain only float values."
    barrierfree = False
    if type(markers) == bool:

        if markers is True:
            barrierfree = True
            markers = marker_list
        else:
            markers = []

    original_index = df.index
    if len(variables) != 0:
        features = variables
    else:

        features = sorted(list(set(df.columns) - set(category)))
    dfpc_list = []
    comb = list(combinations(np.arange(component), 2))

    if len(category) != 0:
        lut = {}
        for i, cat in enumerate(category):
            _clut, _mlut = _create_color_markerlut(df, cat, palette, markers)
            lut[cat] = {"colorlut": _clut, "markerlut": _mlut}

    if len(category) != 0:
        figures = {}
        for cat in category:
            if len(comb) == 1:
                fig, axes = plt.subplots()
                axes = [axes]
            else:
                nrows = len(comb) // 2 + int(len(comb) % 2 != 0)
                if len(figsize) == 0:
                    figsize = [8, 3 * nrows]
                ncols = 2
                fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
                plt.subplots_adjust(top=0.9, right=0.8, left=0.1)
                axes = axes.flatten()
                if len(comb) != ncols * nrows:
                    for i in range(ncols * nrows - len(comb)):
                        fig.delaxes(axes[-(i + 1)])
            figures[cat] = {"fig": fig, "axes": axes}
    else:
        figures = {}
        if len(comb) == 1:
            fig, axes = plt.subplots()
            axes = [axes]
        else:
            nrows = len(comb) // 2 + int(len(comb) % 2 != 0)
            if len(figsize) == 0:
                figsize = [8, 3 * nrows]

            fig, axes = plt.subplots(ncols=2, nrows=nrows, figsize=figsize)
            plt.subplots_adjust(top=0.9, right=0.8, left=0.1)
            axes = axes.flatten()
        figures["nocat"] = {"fig": fig, "axes": axes}
    if method == "pca":
        if regularization:
            x = zscore(x, axis=0)
        pca = PCA(n_components=component, **pcapram)
        pccomp = pca.fit_transform(x)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        combnum = 0
        for axi, (i, j) in enumerate(comb):
            xlabel, ylabel = "pc" + str(i + 1), "pc" + str(j + 1)
            dfpc = pd.DataFrame(
                data=np.array([pccomp[:, i], pccomp[:, j]]).T,
                columns=[xlabel, ylabel],
                index=original_index,
            )
            _loadings = np.array([loadings[:, i], loadings[:, j]]).T
            a = np.sum(_loadings**2, axis=1)
            srtindx = np.argsort(a)[::-1][:arrow_num]
            _loadings = _loadings[srtindx]
            _features = np.array(features)[srtindx]

            if len(category) != 0:
                for cat in category:
                    dfpc[cat] = df[cat]

                    _scatter(
                        dfpc,
                        xlabel,
                        ylabel,
                        cat,
                        figures[cat]["axes"][axi],
                        lut,
                        barrierfree,
                        size,
                        legend=False,
                    )

                    # sns.scatterplot(data=dfpc, x=xlabel, y=ylabel, hue=cat, ax=figures[cat]["axes"][axi],palette=palette)
                    # sns.scatterplot(data=dfpc, x=xlabel, y=ylabel, hue=cat, ax=figures[cat]["axes"][axi],legend=False,palette=palette)
                    if combnum == 1:
                        figures[cat]["axes"][axi].legend(
                            bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0
                        )
                    for k, feature in enumerate(_features):
                        figures[cat]["axes"][axi].arrow(
                            0,
                            0,
                            _loadings[k, 0],
                            _loadings[k, 1],
                            color=arrow_color,
                            width=0.005,
                            head_width=0.1,
                        )
                        figures[cat]["axes"][axi].text(
                            _loadings[k, 0],
                            _loadings[k, 1],
                            feature,
                            color=arrow_text_color,
                        )
            else:
                sns.scatterplot(
                    data=dfpc,
                    x=xlabel,
                    y=ylabel,
                    ax=figures["nocat"][axi],
                    palette=palette,
                )
                # cat=None
                # _scatter(dfpc, xlabel,ylabel, cat, figures["nocat"][axi], lut, barrierfree, size)
                for k, feature in enumerate(_features):
                    # ax.plot([0,_loadings[k, 0] ], [0,_loadings[k, 1] ],color=arrow_color)
                    figures["nocat"][axi].arrow(
                        0,
                        0,
                        _loadings[k, 0],
                        _loadings[k, 1],
                        color=arrow_color,
                        width=0.005,
                        head_width=0.1,
                    )
                    figures["nocat"][axi].text(
                        _loadings[k, 0],
                        _loadings[k, 1],
                        feature,
                        color=arrow_text_color,
                    )

            dfpc_list.append(dfpc)
            combnum += 1

        if len(category) != 0:
            for cat in category:
                figures[cat]["fig"].suptitle(title)
                figures[cat]["fig"].tight_layout(pad=0.5)
                _save(save, cat + "_PCA", fig=figures[cat]["fig"])
        else:
            figures["nocat"]["fig"].suptitle(title)
            figures["nocat"]["fig"].tight_layout(pad=0.5)
            _save(save, "PCA")

        if explained_variance is True:
            fig, ax2 = plt.subplots()
            exp_var_pca = pca.explained_variance_ratio_
            #
            # Cumulative sum of eigenvalues; This will be used to create step plot
            # for visualizing the variance explained by each principal component.
            #
            cum_sum_eigenvalues = np.cumsum(exp_var_pca)
            #
            # Create the visualization plot
            #
            xlabel = ["pc" + str(i + 1) for i in range(0, len(exp_var_pca))]
            plt.bar(
                xlabel,
                exp_var_pca,
                alpha=0.5,
                align="center",
                label="Individual explained variance",
            )
            plt.step(
                range(0, len(cum_sum_eigenvalues)),
                cum_sum_eigenvalues,
                where="mid",
                label="Cumulative explained variance",
            )
            plt.ylabel("Explained variance ratio")
            plt.xlabel("Principal component index")
            _save(save, "ExplainedVar")
        else:
            ax2 = None
        if show is True:
            plt.show()
        return {"data": dfpc_list, "pca": pca, "axes": figures, "axes_explained": ax2}
    elif method == "nmf":
        nmf = NMF(n_components=component, **nmfparam)
        if regularization:
            x = x / np.sum(x, axis=0)[None, :]
        W = nmf.fit_transform(x)
        H = nmf.components_
        combnum = 0
        for axi, (i, j) in enumerate(comb):
            xlabel, ylabel = "p" + str(i + 1), "p" + str(j + 1)
            dfpc = pd.DataFrame(
                data=np.array([W[:, i], W[:, j]]).T,
                columns=[xlabel, ylabel],
                index=original_index,
            )
            if len(category) != 0:
                for cat in category:
                    dfpc[cat] = df[cat]

                    # sns.scatterplot(data=dfpc, x=xlabel, y=ylabel, hue=cat, ax=figures[cat]["axes"][axi],palette=palette)
                    _scatter(
                        dfpc,
                        xlabel,
                        ylabel,
                        cat,
                        figures[cat]["axes"][axi],
                        lut,
                        barrierfree,
                        size,
                        legend=False,
                    )
                    # sns.scatterplot(data=dfpc, x=xlabel, y=ylabel, hue=cat, ax=figures[cat]["axes"][axi],legend=False,palette=palette)
                    if combnum == 1:
                        figures[cat]["axes"][axi].legend(
                            bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0
                        )

            else:
                sns.scatterplot(
                    data=dfpc,
                    x=xlabel,
                    y=ylabel,
                    hue=category,
                    ax=figures["nocat"][axi],
                    palette=palette,
                )
                # cat=None
                # _scatter(dfpc, xlabel,ylabel, cat, figures["nocat"][axi], lut, barrierfree, size)
            dfpc_list.append(dfpc)
            combnum += 1
        if len(category) != 0:
            for cat in category:
                figures[cat]["fig"].suptitle(title)
                figures[cat]["fig"].tight_layout(pad=0.5)
                _save(save, cat + "_NMF", fig=figures[cat]["fig"])
        else:
            figures["nocat"]["fig"].suptitle(title)
            figures["nocat"]["fig"].tight_layout(pad=0.5)
        _save(save, "NMF")
        if explained_variance is True:
            fig, axes2 = plt.subplots(nrows=component, figsize=[5, 5])
            axes2 = axes2.flatten()
            for i, ax in enumerate(axes2):
                if i == 0:
                    ax.set_title("Coefficients of matrix H")
                ax.bar(np.arange(len(features)), H[i])
                ax.set_ylabel("p" + str(i + 1))
                ax.set_xticks(np.arange(len(features)), labels=[])
            ax.set_xticks(np.arange(len(features)), labels=features, rotation=90)
            fig.tight_layout()

            # dfw={"index":[],"p":[],"val":[]}
            # ps=["p"+str(i+1) for i in range(component)]
            # originalindex=df.index
            # for i in range(W.shape[0]):
            #     for j in range(W.shape[1]):
            #         dfw["index"].append(originalindex[i])
            #         dfw["p"].append(ps[j])
            #         dfw["val"].append(W[i,j])
            # dfw=pd.DataFrame(data=dfw)
            #
            # dfh={"feature":[],"p":[],"val":[]}
            # for i in range(H.shape[0]):
            #     for j in range(H.shape[1]):
            #         dfh["p"].append(ps[i])
            #         dfh["feature"].append(features[j])
            #
            #         dfh["val"].append(H[i,j])
            # dfw=pd.DataFrame(data=dfw)
            # dfh=pd.DataFrame(data=dfh)
            # #dotplot(dfw,row="index",col="p",size_val="val")
            # dotplot(dfh,row="p",col="feature",size_val="val",)
            _save(save, "Coefficients")
        else:
            axes2 = None
        if show is True:
            plt.show()
        return {
            "data": dfpc_list,
            "W": W,
            "H": H,
            "axes": figures,
            "axes_explained": axes2,
        }
    elif method == "lda":
        _ = LatentDirichletAllocation(n_components=component, random_state=0)
        if regularization:
            x = x / np.sum(x, axis=0)[None, :]

    else:
        raise Exception(
            "{} is not in options. Available options are: pca, nmf".format(method)
        )


def manifoldplot(
    df: pd.DataFrame,
    variables: List = [],
    category: Union[List, str] = "",
    method: str = "tsne",
    show: bool = False,
    figsize=[5, 5],
    title: str = "",
    param: dict = {},
    save: str = "",
    ax: Optional[plt.Axes] = None,
    palette="tab20c",
    markers: bool = False,
    size: float = 10,
    **kwargs,
):
    """
    Reducing the dimensionality of data and drawing a scatter plot.

    Parameters
    ----------
    df : pandas DataFrame
    category: list or str, optional
        the column name of a known sample category (if exists).
    method: str
        Method name for decomposition.
        Available methods: {"random_projection": "Sparse random projection",
                            "linear_discriminant": "Linear discriminant analysis",
                            "isomap": "Isomap",
                            "lle": "Locally linear embedding",
                            "modlle": "Modified locally linear embedding",
                            "hessian_lle":" Hessian locally linear embedding",
                            "ltsa_lle": "LTSA",
                            "mds": "MDS",
                            "random_trees": "Random Trees Embedding",
                            "spectral": "Spectral embedding",
                            "tsne": "TSNE",
                            "nca": "Neighborhood components analysis",
                            "umap":"UMAP"}
    component: int
        The number of components
    n_neighbors: int
        The number of neighbors related to isomap and lle methods.

    show : bool
        Whether or not to show the figure.

    Returns
    -------
    Raises
    ------
    Notes
    -----
    References
    ----------
    See Also
    --------
    Examples
    --------
    """
    method_dict = {
        "random_projection": "Sparse random projection",
        "linear_discriminant": "Linear discriminant analysis",
        "isomap": "Isomap",
        "lle": "Locally linear embedding",
        "modlle": "Modified locally linear embedding",
        "hessian_lle": " Hessian locally linear embedding",
        "ltsa_lle": "LTSA",
        "mds": "MDS",
        "random_trees": "Random Trees Embedding",
        "spectral": "Spectral embedding",
        "tsne": "TSNE",
        "nca": "Neighborhood components analysis",
        "umap": "UMAP",
    }
    x, category = _separate_data(df, variables=variables, category=category)
    barrierfree = False
    if type(markers) == bool:

        if markers is True:
            barrierfree = True
            markers = marker_list
        else:
            markers = []
    if len(category) != 0:
        lut = {}
        for i, cat in enumerate(category):
            _clut, _mlut = _create_color_markerlut(df, cat, palette, markers)
            lut[cat] = {"colorlut": _clut, "markerlut": _mlut}

    x = zscore(x, axis=0)
    # features = df.columns
    original_index = df.index
    embedding = _get_embedding(method=method, param=param)
    Xt = embedding.fit_transform(x)
    dft = pd.DataFrame(
        data=np.array([Xt[:, 0], Xt[:, 1]]).T,
        columns=["d1", "d2"],
        index=original_index,
    )

    if len(category) != 0:
        figsize = [5 * len(category), 5]
        fig, axes = plt.subplots(figsize=figsize, ncols=len(category))
        axes = axes.flatten()
        for cat, ax in zip(category, axes):
            dft[cat] = df[cat]
            _scatter(dft, "d1", "d2", cat, ax, lut, barrierfree, size)
            # sns.scatterplot(data=dft, x="d1", y="d2", hue=cat, ax=ax,palette=palette,**kwargs)
    else:
        fig, axes = plt.subplots(figsize=figsize)
        sns.scatterplot(data=dft, x="d1", y="d2", ax=axes, palette=palette, **kwargs)
    if title != "":
        fig.suptitle(title)
    else:
        fig.suptitle(method_dict[method])
    if show is True:
        plt.show()
    _save(save, method_dict[method])
    return {"data": dft, "axes": axes}


def regression_single(
    df: pd.DataFrame,
    x: str,
    y: str,
    method: str = "ransac",
    category: str = "",
    figsize: List[int] = [5, 5],
    show=False,
    ransac_param={"max_trials": 1000},
    robust_param: dict = {},
    xunit: str = "",
    yunit: str = "",
    title: str = "",
    random_state: int = 42,
    ax: Optional[plt.Axes] = None,
    save: str = "",
) -> Dict:
    """
    Drawing a scatter plot with a single variable linear regression.

    Parameters
    ----------
    df : pandas DataFrame

    x: str
        the column name of x axis.
    y: str
        the column name of y axis.

    method: str
        Method name for regression. Default: ransac
        Available methods: ["ransac",
                            "robust",
                            "lasso","elastic_net"
                            ]
    figsize: list[int]
        figure size
    show : bool
        Whether or not to show the figure.
    robust_param: dict, optional
        Hyper parameters for robust regression. Please see https://www.statsmodels.org/dev/generated/statsmodels.robust.robust_linear_model.RLM.html
    xunit: str, optional
        X axis unit
    yunit: str, optional
        Y axis unit
    title: str, optional
        Figure title

    random_state: int, optional (default=42)
        random state for RANSAC regression
    ax: plt.Axes, optional,
        pyplot axis
    save: str, optional
        The prefix of file names to save.
    Returns
    -------
    dict: dict {"axes":ax, "coefficient":coef,"intercept":intercept,"coefficient_pval":coef_p, "r2":r2, "fitted_model":fitted_model}

        fitted_model:
            this can be used like: y_predict=fitted_model.predict(_X)
    Raises
    ------
    Notes
    -----
    References
    ----------
    See Also
    --------
    Examples
    --------
    """

    Y = df[y]
    X = np.array(df[x])
    plotline_X = np.arange(X.min(), X.max()).reshape(-1, 1)

    plt.rcParams.update({"font.size": 14})
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(title)
    else:
        fig = None
        # plt.title(title)
    plt.subplots_adjust(left=0.15)
    if method == "ransac":
        _title = "RANSAC regression, r2: {:.2f}, MSE: {:.2f}\ny = {:.2f} + {:.2f}x, coefficient p-value: {:.2E}"
        (
            fitted_model,
            coef,
            coef_p,
            intercept,
            r2,
            x_line,
            y_line,
            ci,
            pi,
            std_error,
            MSE,
            inlier_mask,
            outlier_mask,
        ) = _ransac(X, Y, plotline_X, random_state, ransac_param)

        _draw_ci_pi(ax, ci, pi, x_line, y_line)
        sns.scatterplot(
            x=X[inlier_mask], y=Y[inlier_mask], color="blue", label="Inliers"
        )
        sns.scatterplot(
            x=X[outlier_mask], y=Y[outlier_mask], color="red", label="Outliers"
        )
        plt.xlabel(x)
        plt.ylabel(y)
        # print(r2, MSE,ransac_coef,ransac.estimator_.intercept_)
        plt.title(_title.format(r2, MSE, coef, intercept, coef_p))
        plt.plot(plotline_X.flatten(), y_line)

        _save(save, method)
        if len(category) != 0:
            fig, ax1 = plt.subplots(figsize=figsize)
            plt.subplots_adjust(left=0.15)
            _draw_ci_pi(ax1, ci, pi, x_line, y_line)
            sns.scatterplot(data=df, x=x, y=y, hue=category, ax=ax1)

            plt.xlabel(x)
            plt.ylabel(y)
            # print(r2, MSE,ransac_coef,ransac.estimator_.intercept_)
            plt.title(_title.format(r2, MSE, coef, intercept, coef_p))
            plt.plot(plotline_X.flatten(), y_line)
            _save(save, method + "_" + category)
    elif method == "robust":
        _title = "Robust linear regression, r2: {:.2f}, MSE: {:.2f}\ny = {:.2f} + {:.2f}x , p-values: coefficient {:.2f}, \
        intercept {:.2f}"
        (
            fitted_model,
            summary,
            coef,
            coef_p,
            intercept,
            intercept_p,
            r2,
            x_line,
            y_line,
            ci,
            pi,
            std_error,
            MSE,
        ) = _robust_regression(X, Y, plotline_X, robust_param)

        _draw_ci_pi(ax, ci, pi, x_line, y_line)
        sns.scatterplot(data=df, x=x, y=y, color="blue")
        # print(r2, MSE,ransac_coef,ransac.estimator_.intercept_)
        plt.title(_title.format(r2, MSE, coef, intercept, coef_p, intercept_p))
        plt.plot(plotline_X.flatten(), y_line)
        _save(save, method)
        if len(category) != 0:
            fig, ax1 = plt.subplots(figsize=figsize)
            plt.subplots_adjust(left=0.15)
            _draw_ci_pi(ax1, ci, pi, x_line, y_line)
            sns.scatterplot(data=df, x=x, y=y, hue=category, ax=ax1)
            # print(r2, MSE,ransac_coef,ransac.estimator_.intercept_)
            plt.title(_title.format(r2, MSE, coef, intercept, coef_p, intercept_p))
            plt.plot(plotline_X.flatten(), y_line)
            _save(save, method + "_" + category)
    elif method == "lasso" or method == "elastic_net" or method == "ols":
        _title = "OLS ({}), r2: {:.2f}, MSE: {:.2f}\ny = {:.2f} + {:.2f}x, coefficient p-value: {:.2E}"

        (
            fitted_model,
            coef,
            coef_p,
            intercept,
            r2,
            x_line,
            y_line,
            ci,
            pi,
            std_error,
            MSE,
        ) = _ols(X, Y, plotline_X, method)

        _draw_ci_pi(ax, ci, pi, x_line, y_line)
        sns.scatterplot(data=df, x=x, y=y, color="blue")
        # print(r2, MSE,ransac_coef,ransac.estimator_.intercept_)
        plt.title(_title.format(method, r2, MSE, coef, intercept, coef_p))
        plt.plot(plotline_X.flatten(), y_line)
        _save(save, method)
        if len(category) != 0:
            fig, ax1 = plt.subplots(figsize=figsize)
            plt.subplots_adjust(left=0.15)
            _draw_ci_pi(ax1, ci, pi, x_line, y_line)
            sns.scatterplot(data=df, x=x, y=y, color="blue", hue=category)
            # print(r2, MSE,ransac_coef,ransac.estimator_.intercept_)
            plt.title(_title.format(method, r2, MSE, coef, intercept, coef_p))
            plt.plot(plotline_X.flatten(), y_line)
            _save(save, method + "_" + category)
    if yunit != "":
        ax.text(0, 1, "({})".format(yunit), transform=ax.transAxes, ha="right")
        ax1.text(0, 1, "({})".format(yunit), transform=ax.transAxes, ha="right")
    if xunit != "":
        ax.text(1, 0, "({})".format(xunit), transform=ax.transAxes, ha="left", va="top")
        ax1.text(
            1, 0, "({})".format(xunit), transform=ax.transAxes, ha="left", va="top"
        )

    return {
        "axes": ax,
        "coefficient": coef,
        "intercept": intercept,
        "coefficient_pval": coef_p,
        "r2": r2,
        "fitted_model": fitted_model,
    }


def _robust_regression(X, Y, plotline_X, robust_param):
    n = X.shape[0]
    rlm_model = sm.RLM(
        Y, sm.add_constant(X), M=sm.robust.norms.HuberT(), **robust_param
    )
    fitted_model = rlm_model.fit()
    summary = fitted_model.summary()
    coef = fitted_model.params[1]
    intercept = fitted_model.params[0]
    intercept_p = fitted_model.pvalues[0]
    coef_p = fitted_model.pvalues[1]
    y_model = fitted_model.predict(sm.add_constant(X))
    r2 = _calc_r2(X, Y)
    x_line = plotline_X.flatten()
    y_line = fitted_model.predict(sm.add_constant(x_line))

    ci, pi, std_error = _ci_pi(X, Y, plotline_X.flatten(), y_model)
    MSE = 1 / n * np.sum((Y - y_model) ** 2)
    return (
        fitted_model,
        summary,
        coef,
        coef_p,
        intercept,
        intercept_p,
        r2,
        x_line,
        y_line,
        ci,
        pi,
        std_error,
        MSE,
    )


def _ols(X, Y, plotline_X, method):
    n = X.shape[0]
    if method == "lasso":
        method = "sqrt_lasso"
    rlm_model = sm.OLS(Y, sm.add_constant(X))
    if method == "ols":
        fitted_model = rlm_model.fit()
    else:
        fitted_model = rlm_model.fit_regularized(method)
    coef = fitted_model.params[1]
    intercept = fitted_model.params[0]
    y_model = fitted_model.predict(sm.add_constant(X))
    r2 = _calc_r2(X, Y)
    x_line = plotline_X.flatten()
    y_line = fitted_model.predict(sm.add_constant(x_line))
    ci, pi, std_error = _ci_pi(X, Y, plotline_X.flatten(), y_model)
    q = (X - X.mean()).transpose() @ (X - X.mean())
    sigma = std_error * (q**-1) ** (0.5)
    # print(sigma,coef )
    coef_p = stats.t.sf(abs(coef / sigma), df=X.shape[0] - 2)
    MSE = 1 / n * np.sum((Y - y_model) ** 2)
    return (
        fitted_model,
        coef,
        coef_p,
        intercept,
        r2,
        x_line,
        y_line,
        ci,
        pi,
        std_error,
        MSE,
    )


def _ransac(X, Y, plotline_X, random_state, ransac_param):
    n = X.shape[0]
    _X = np.array(X).reshape([-1, 1])
    fitted_model = RANSACRegressor(random_state=random_state, **ransac_param).fit(_X, Y)
    y_line = fitted_model.predict(plotline_X)
    coef = fitted_model.estimator_.coef_[0]
    intercept = fitted_model.estimator_.intercept_
    inlier_mask = fitted_model.inlier_mask_
    outlier_mask = np.invert(inlier_mask)

    # number of samples
    y_model = fitted_model.predict(_X)

    r2 = _calc_r2(X, Y)
    # mean squared error
    MSE = 1 / n * np.sum((Y - y_model) ** 2)

    # to plot the adjusted model
    x_line = plotline_X.flatten()

    ci, pi, std_error = _ci_pi(X, Y, plotline_X.flatten(), y_model)
    q = (X - X.mean()).transpose() @ (X - X.mean())
    sigma = std_error * (q**-1) ** (0.5)
    coef_p = stats.t.sf(
        abs(fitted_model.estimator_.coef_[0] / sigma), df=X.shape[0] - 2
    )

    return (
        fitted_model,
        coef,
        coef_p,
        intercept,
        r2,
        x_line,
        y_line,
        ci,
        pi,
        std_error,
        MSE,
        inlier_mask,
        outlier_mask,
    )


def _regression(df, x, y, ax, cat="", _clut={}, robust_param={}, newkey="", color=""):
    reg_legend_elements = []
    fitted_models = {}
    reg_res = {}
    if cat != "":

        for key in _clut.keys():
            key_filt = df[cat] == key
            rdf = df.loc[key_filt]
            rX, rY = rdf[x], rdf[y]
            plotline_X = np.arange(rX.min(), rX.max()).reshape(-1, 1)
            (
                fitted_model,
                summary,
                coef,
                coef_p,
                intercept,
                intercept_p,
                r2,
                x_line,
                y_line,
                ci,
                pi,
                std_error,
                MSE,
            ) = _robust_regression(rX, rY, plotline_X, robust_param)
            _tmpcolor = np.array(_clut[key])
            _draw_ci_pi(
                ax,
                ci,
                pi,
                x_line,
                y_line,
                pi_color=_tmpcolor,
                ci_color=_tmpcolor + (1 - _tmpcolor) * 0.5,
                alpha=0.75,
            )
            ax.plot(x_line, y_line, c=_tmpcolor)
            reg_legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="",
                    linewidth=3,
                    color=_tmpcolor,
                    label="\u03B21: {x:.2f}\np: {p:.1E}\nr2: {y:.2f}".format(
                        x=coef, y=r2, p=coef_p
                    ),
                )
            )
            fitted_models[cat + "_" + key] = fitted_model
            reg_res[cat + "_" + key] = {
                "coefficient": coef,
                "coefficient_pvalue": coef_p,
                "intercept": intercept,
                "intercept_pvalue": intercept_p,
                "r2": r2,
                "mse": MSE,
            }
    else:

        rX, rY = df[x], df[y]
        plotline_X = np.arange(rX.min(), rX.max()).reshape(-1, 1)
        (
            fitted_model,
            summary,
            coef,
            coef_p,
            intercept,
            intercept_p,
            r2,
            x_line,
            y_line,
            ci,
            pi,
            std_error,
            MSE,
        ) = _robust_regression(rX, rY, plotline_X, robust_param)
        if color != "":
            _tmpcolor = np.array(mcolors.to_rgb(color))
        else:
            _tmpcolor = np.array([0, 0.75, 0])
        _draw_ci_pi(
            ax,
            ci,
            pi,
            x_line,
            y_line,
            pi_color=_tmpcolor,
            ci_color=_tmpcolor + (1 - _tmpcolor) * 0.5,
            alpha=0.75,
        )
        ax.plot(x_line, y_line, c=_tmpcolor)
        reg_legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="",
                linewidth=3,
                color=_tmpcolor,
                label="\u03B21: {x:.2f}\np: {p:.1E}\nr2: {y:.2f}".format(
                    x=coef, y=r2, p=coef_p
                ),
            )
        )
        if newkey == "":
            newkey = "regression"
        fitted_models[newkey] = fitted_model
        reg_res[newkey] = {
            "coefficient": coef,
            "coefficient_pvalue": coef_p,
            "intercept": intercept,
            "intercept_pvalue": intercept_p,
            "r2": r2,
            "mse": MSE,
        }
    ax.add_artist(
        ax.legend(
            handles=reg_legend_elements,
            title="Regression",
            loc="upper left",
            bbox_to_anchor=(0.0, -0.1),
            ncol=3,
        )
    )
    return fitted_models, reg_res


def volcanoplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    label: str = "",
    logscalex: bool = False,
    logscaley: bool = True,
    sizes: str = "",
    xthreshold: float = 1,
    ythreshold: float = 0.05,
    topn_labels: int = 5,
    rankby: str = "both",
    topn_labels_left: int = 0,
    topn_labels_right: int = 0,
    box: bool = True,
    base_color: str = "gray",
    highlight_color: str = "red",
    save: str = "",
    write_csv: Union[bool, str] = False,
    ax: Optional[plt.Axes] = None,
    fig: Optional[mpl.figure.Figure] = None,
    markers: bool = False,
    size: float = 5.0,
    size_scale: float = 100,
    plotline: bool = True,
    linestyle="-",
    linecolor="darkcyan",
    alpha: float = 1,
    size_format: str = "",
    xformat: str = "",
    yformat: str = "",
    xunit: str = "",
    yunit: str = "",
    size_unit: str = "",
    title: str = "",
    figsize: list = [],
    gridspec_kw: dict = {},
):

    if topn_labels_left == 0 and topn_labels_right == 0:
        topn_labels_left = topn_labels
        topn_labels_right = topn_labels

    df = copy.deepcopy(df)
    if len(figsize) == 0:
        figsize = [5, 5]
    if ax is None:

        fig, ax = plt.subplots(figsize=figsize)

    if np.amax(df[y]) <= 1:

        if logscaley is False:
            print(
                "Transforming {} to -log10 values. if you do not want it, set 'logscaley=False'.".format(
                    y
                )
            )
            df[y] = -np.log10(df[y])
    elif logscaley is True:
        df[y] = -np.log10(df[y])
    if logscalex is True:
        df[x] = np.log2(df[x])
    if ythreshold < 1:
        ythreshold = -np.log10(ythreshold)
    updown = []
    for _x, _y in zip(list(df[x]), list(df[y])):
        if (_y <= ythreshold) or (np.abs(_x) < xthreshold):
            updown.append("ns")
        elif _y > ythreshold and _x >= xthreshold:
            updown.append("up")
        elif _y > ythreshold and _x <= -xthreshold:
            updown.append("down")
    df["updown"] = updown

    sig = df.loc[df["updown"] == "up"]
    sig = sig.reset_index()
    sigm = df.loc[df["updown"] == "down"]
    sigm = sigm.reset_index()
    palette = {
        "ns": mcolors.to_rgb(base_color),
        "up": mcolors.to_rgb(highlight_color),
        "down": mcolors.to_rgb(highlight_color),
    }

    if sizes != "":

        scatterplot(
            df=df,
            x=x,
            y=y,
            category="updown",
            ax=ax,
            sizes=sizes,
            color=base_color,
            alpha=alpha,
            edgecolors=None,
            show_legend=False,
            size_format=size_format,
            xunit=xunit,
            yunit=yunit,
            size_unit=size_unit,
            size_scale=size_scale,
            markers=markers,
            xformat=xformat,
            yformat=yformat,
        )
        # ax.scatter(nonsig[x], nonsig[y], c=base_color, s=nonsig[sizes], alpha=alpha,)
        # scatterplot(df=sig, x=x, y=y, ax=ax,sizes=sizes, color=highlight_color, alpha=alpha, edgecolors=None, show_legend=False)
        # scatterplot(df=sigm, x=x, y=y, ax=ax,sizes=sizes, color=highlight_color, alpha=alpha, edgecolors=None, show_legend=False, xunit=xunit, yunit=yunit, title=title)
        # # ax.scatter(sig[x], sig[y], c=highlight_color, s=sig[sizes])
        # ax.scatter(sigm[x], sigm[y], c=highlight_color, s=sigm[sizes])
    else:
        scatterplot(
            df=df,
            x=x,
            y=y,
            category="updown",
            palette=palette,
            ax=ax,
            size=size,
            color=base_color,
            alpha=alpha,
            edgecolors=None,
            show_legend=False,
            xunit=xunit,
            yunit=yunit,
            markers=markers,
            xformat=xformat,
            yformat=yformat,
        )
        # scatterplot(df=nonsig, x=x, y=y, ax=ax,size=size, color=base_color, alpha=alpha, edgecolors=None, show_legend=False)
        # scatterplot(df=sig, x=x, y=y, ax=ax,size=size, color=highlight_color, alpha=alpha, edgecolors=None, show_legend=False)
        # scatterplot(df=sigm, x=x, y=y, ax=ax,size=size, color=highlight_color, alpha=alpha, edgecolors=None, show_legend=False, xunit=xunit, yunit=yunit, title=title)
        # ax.scatter(nonsig[x], nonsig[y], c=base_color, s=size)
        # ax.scatter(sig[x], sig[y], c=highlight_color, s=size)
        # ax.scatter(sigm[x], sigm[y], c=highlight_color, s=size)

    # ax.set_xlabel(x)
    # ax.set_xlabel(y)

    xsrt = np.argsort(sig[x])[::-1]
    xrank = np.argsort(xsrt)
    xsrtm = np.argsort(np.abs(sigm[x]))[::-1]
    xrankm = np.argsort(xsrtm)

    ysrt = np.argsort(sig[y])[::-1]
    yrank = np.argsort(ysrt)
    ysrtm = np.argsort(sigm[y])[::-1]
    yrankm = np.argsort(ysrtm)
    if rankby == "both":
        rank = (xrank + yrank) / 2
        rankm = (xrankm + yrankm) / 2
    elif rankby == "x" or rankby == x:
        rank = xrank
        rankm = xrankm
    elif rankby == "y" or rankby == y:
        rank = yrank
        rankm = yrankm
    labeled_genes = []
    texts = []
    for _rank, _sig, _topn_labels in zip(
        [rank, rankm], [sig, sigm], [topn_labels_right, topn_labels_left]
    ):
        if len(_rank) != 0:
            top_index = np.argsort(_rank)[:_topn_labels]
            topsig = _sig.iloc[top_index]
            if label == "":
                labels = topsig.index
            else:
                labels = topsig[label]

            for _x, _y, _l in zip(topsig[x], topsig[y], labels):
                # ax.text(_x, _y, _l)
                if box is True:
                    texts.append(
                        ax.text(
                            _x,
                            _y,
                            _l,
                            va="bottom",
                            bbox=dict(
                                boxstyle="round,pad=0.0",
                                fc="white",
                                ec="y",
                                lw=0.5,
                                alpha=0.9,
                            ),
                        )
                    )
                else:
                    texts.append(ax.text(_x, _y, _l, va="bottom"))
            labeled_genes.append(topsig)
    adjust_text(
        texts,
        arrowprops=dict(arrowstyle="-", color="black", lw=0.5),
        force_text=(0.2, 0.4),
    )
    _, ymax = np.amin(df[y]), np.amax(df[y])
    xmin, xmax = np.amin(df[x]), np.amax(df[x])

    if plotline is True:
        ax.plot(
            [xthreshold, xthreshold],
            [ythreshold, ymax],
            linestyle,
            color=linecolor,
            alpha=0.5,
        )
        ax.plot(
            [-xthreshold, -xthreshold],
            [ythreshold, ymax],
            linestyle,
            color=linecolor,
            alpha=0.5,
        )
        ax.plot(
            [xthreshold, xmax],
            [ythreshold, ythreshold],
            linestyle,
            color=linecolor,
            alpha=0.5,
        )
        ax.plot(
            [xmin, -xthreshold],
            [ythreshold, ythreshold],
            linestyle,
            color=linecolor,
            alpha=0.5,
        )
        # ax.plot([xmin, xmax], [ythreshold, ythreshold],linestyle)
    if title != "":
        if fig is None:
            fig.suptitle(title)
        else:
            plt.title(title)
    _save(save, "volcano")
    res = {"upgenes": sig, "downgenes": sigm, "labeledgenes": pd.concat(labeled_genes)}
    if type(write_csv) == bool and write_csv is True:
        for k, v in res.items():
            v.to_csv(k + ".csv")
    elif type(write_csv) == str and write_csv != "":
        for k, v in res.items():
            v.to_csv(write_csv + "_" + k + ".csv")
    return res


def manhattanplot(
    df: pd.DataFrame,
    x: str = "BP",
    p: str = "P",
    chrom: str = "CHR",
    snp: str = "SNP",
    zscore: str = "",
    effectsize: str = "",
    gene: str = "",
    distance: str = "",
    logtransform: bool = True,
    threshold: float = 4,
    thresholdcolor: str = "red",
):
    df = df.reset_index()
    gb = df.groupby(chrom)
    if logtransform is True:

        df[p] = -np.log10(df[p])
    if threshold < 1:
        threshold = -np.log10(threshold)
    colors = ["gray", "black"]

    fig, ax = plt.subplots()

    for i, x in enumerate(gb.groups):

        _df = gb.get_group(x)
        ax.scatter(_df.index, _df[p], c=colors[i % 2], s=5)
        ax.text((np.max(_df.index) + np.min(_df.index)) / 2, -0.5, x)
    _df = df.loc[df[p] > threshold]
    ax.scatter(_df.index, _df[p], c=thresholdcolor, s=4)
    maxindex = np.argmax(df[p])
    ax.text(
        df.index[maxindex], df[p][maxindex], df[snp][maxindex]
    )  # +"\n"+str(df[chrom][maxindex])+":"+str(df[x][maxindex]))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.set_xticks([])
