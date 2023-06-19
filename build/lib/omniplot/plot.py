"""
A main plotting module for omniplot. 
"""
import copy
from typing import Union, Optional, Dict, List
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore
import scipy.stats as stats
from matplotlib.ticker import StrMethodFormatter
from omniplot.scatter import *
from omniplot.proportion import *
from omniplot.heatmap import *
from omniplot.utils import colormap_list, hatch_list, marker_list, linestyle_list
from omniplot.utils import _save, _separate_data,  _dendrogram_threshold, _radialtree2,_get_cluster_classes, _draw_ci_pi #
from omniplot.scatter import _robust_regression

# hatch_list: list = ['//', '\\\\', '||', '--', '++', 'xx', 'oo', 'OO', '..', '**','/o', '\\|', '|*', '-\\', '+o', 'x*', 'o-', 'O|', 'O.', '*-']
# marker_list: list=['.', '_' , '+','|', 'x', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'D', 'd', 'P', 'X','o', '1', '2', '3', '4','|', '_']

plt.rcParams['font.family']= 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['svg.fonttype'] = 'none'
sns.set_theme(font="Arial")

def radialtree(df: pd.DataFrame,
               n_clusters: int=3,
               x: str="",
               variables: List=[],
               category: Union[str, List[str]]=[],
               ztransform: bool=True,
               save: str="",
               distance_method="euclidean",
               tree_method="ward",
               title: str="",
               y: list=[],
               linewidth: float=1,
               figsize: Optional[list]=None,
               **kwargs) -> Dict:
    """
    Drawing a radial dendrogram with color labels.
    
    Parameters
    ----------
    df : pandas DataFrame
        A wide format data. 
        
        
    n_clusters: int, optional (default: 3)
        Approximate number of clusters to produce
    x: str, optional
        the name of columns containing sample names. If not provided, the index will be considered sample names.
    variables: list, optional
        the name of columns containing variables to calculate the distances between samples
    category: str or list of str
        the column name of a category that is going to presented as colors around the dendrogram.
    ztransform: bool=True,
    save: str="",
    distance_method="euclidean",
    tree_method="ward",
    
    
    show : bool
        Whether or not to show the figure.
    fontsize : float
        A float to specify the font size
    figsize : [x, y] array-like
        1D array-like of floats to specify the figure size
    palette : string
        Matlab colormap name.
    Returns
    -------
    dict: {"axes":ax, "clusters": clusters}

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
    if len(y)!=0:
        variables=y
    
    if x !="":
        _labels=df[x]
        df=df.drop(x, axis=1)
    else:
        _labels=df.index

    X, category=_separate_data(df, variables=variables, category=category)
    category_df=df[category]

    if ztransform==True:
        X=zscore(X, axis=0)

    D=squareform(pdist(X,metric=distance_method))
    Y = sch.linkage(D, method=tree_method)
    
    Z = sch.dendrogram(Y,labels=_labels,no_plot=True)
    t=_dendrogram_threshold(Z, n_clusters)
    _fig, _ax =plt.subplots()
    Z=sch.dendrogram(Y,
                        labels = _labels,
                        color_threshold=t, ax=_ax)
    #xticks=set(_ax.get_xticks())
    plt.close(_fig)
    sample_classes={k: list(category_df[k]) for k in category_df.columns}
    ax=_radialtree2(Z, 
                    sample_classes=sample_classes,
                    addlabels=False,
                    linewidth=linewidth,
                    figsize=figsize, **kwargs)
    if title !="":
        ax.set_title(title)
    _save(save, "radialtree")
    clusters = _get_cluster_classes(Z)
    return {"axes":ax, "clusters":clusters}


def violinplot(*args, **kwargs):
    catplot(*args, **kwargs)

def catplot(df: pd.DataFrame, 
               x: str, 
               y: str,
               pairs: list=[], 
               test: str="ttest_ind",
               alternative: str="two-sided",
               significance: str="numeric",
               significance_ranges: Dict[str, float]={"*":-np.log10(0.05),"**":4,"***":10},
               swarm: bool=False,
               xorder: list=[],
               equal_var: bool=False, 
               yunit: str="",
               title: str="",
               save: str="",
               inner: str="quartile",
               ax: Optional[plt.Axes]=None,
               plotkw: dict={},
               violinkw: dict={},
               color: str="lightgray",
               kind: str="violin",
               points: str="",
               **kwargs)->Dict:
    """
    Draw a boxplot with a statistical test 
    
    Parameters
    ----------
    df : pandas DataFrame
    
    x,y: str
        names of variables in data
    pairs: list, optional
        Category pairs for the statistical test.
        Examples: [["Adelie","Chinstrap" ],
                    ["Gentoo","Chinstrap" ],
                    ["Adelie","Gentoo" ]]
    test: str, optional
        Method name for the statistical test. Defalt: ttest_ind
        Available methods: ["ttest_ind",
                            "ttest_rel",
                            "kruskal",
                            "mannwhitneyu",
                            "wilcoxon",
                            "brunnermunzel",
                            "median_test"]
    alternative: str ['two-sided', 'less', 'greater'], optional
        Defines the alternative hypothesis. Defalt: "two-sided"
    
    show : bool, optional
        Whether or not to show the figure.
    significance: str ['numeric', 'symbol'], optional
        How to show the significance. 'numeric' will show -log10(p values) in the plot and 
        'symbol' will represent significance as asterisks.
    significance_ranges: dict, optional 
        thresholds of -log10(p values) that each asterisk number represents. Ignored when  significance="numeric".
        example: {"*":-np.log10(0.05),"**":4,"***":10}
    swarm: bool, optional
        Whether or not to superpose a swarm plot. Not recommended if the sample size is too large.
    xorder: list, optional
        The order of x axis labels
    equal_var: bool, optional
        Related to ttest_ind method. The default is True, which will produce a p value equal to t-test in R.
    kind: str ["violin", "box", lines], optional (defalt: violin)
        The kind of the plot to draw.
    points: str ["swarm", "strip"], optional (defalt: swarm)
        The style of the swarm plot.
    kwargs: any options accepted by scipy statistical test functions
    
     
    Returns
    -------
    dict("p values":pvalues,"axes":ax)
    
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
    tests=["ttest_ind","ttest_rel","kruskal","mannwhitneyu","wilcoxon","brunnermunzel","median_test"]
    if test not in tests:
        raise ValueError("Available tests are "+", ".join(tests))
    
    if len(xorder)==0:
        xorder=sorted(list(set(df[x])))
    pvals=[]
    for p1,p2 in pairs:
        
        statstest=getattr(stats, test)
        if test=="wilcoxon" or test=="ttest_rel":
            _, pval,_=statstest(df[y][df[x]==p1],df[y][df[x]==p2],alternative=alternative,**kwargs)
        elif test=="median_test":
            _, pval,_,_=statstest(df[y][df[x]==p1],df[y][df[x]==p2],alternative=alternative,**kwargs)
        elif test=="ttest_ind":
            _, pval=statstest(df[y][df[x]==p1],df[y][df[x]==p2],alternative=alternative,equal_var=equal_var,**kwargs)
        
        else:
            _, pval=statstest(df[y][df[x]==p1],df[y][df[x]==p2],alternative=alternative,**kwargs)
        
        p1ind=xorder.index(p1)
        p2ind=xorder.index(p2)
        if pval==0:
            pval=np.inf
        else:
            pval=-np.log10(pval)
        pvals.append([np.abs(p2ind-p1ind), np.amin([p2ind, p1ind]),np.amax([p2ind, p1ind]), pval])
    pvals = sorted(pvals, key = lambda x: (x[0], x[1]))
    if isinstance(ax, type(None)):
        fig, ax=plt.subplots(**plotkw)
    else:
        fig=None
    if kind=="violin":
        sns.violinplot(data=df, x=x,y=y, order=xorder,inner=inner, ax=ax, color=color, **violinkw)
    elif kind=="box":
        sns.boxenplot(data=df, x=x,y=y, order=xorder, ax=ax, color=color, **violinkw)
    elif kind=="lines":
        for i, _x in enumerate(xorder):
            val=df[y][df[x]==_x].values
            q95, q75, q50, q25, q5=np.quantile(val, [0.95, 0.75, 0.5, 0.25, 0.05])
            ax.plot([i,i],[q5, q95], color="black")
            ax.plot([i-0.125,i+0.125],[q5, q5], color="black")
            ax.plot([i-0.125,i+0.125],[q95, q95], color="black")
            ax.plot([i-0.25,i+0.25],[q25, q25], color="black")
            ax.plot([i-0.25,i+0.25],[q75, q75], color="black")
            ax.plot([i-0.45,i+0.45],[q50, q50], color="black")

    if swarm is True or points=="swarm":
        sns.swarmplot(data=df, x=x,y=y, order=xorder,color="black",alpha=0.5, ax=ax)
    elif points=="strip":
        sns.stripplot(data=df, x=x,y=y, order=xorder,color="black",alpha=0.5, ax=ax)


    ymax=np.amax(df[y])
    ymin=np.amin(df[y])
    newpvals={}
    for i, pval in enumerate(pvals):
        ax.plot([pval[1],pval[2]], [ymax+(ymax-ymin)*(0.05+i*0.051),ymax+(ymax-ymin)*(0.05+i*0.051)], color="black")
        p=np.round(pval[-1],2)
        
        newpvals[xorder[pval[1]]+"_"+xorder[pval[2]]]=p
        if significance=="numeric":
            if p<=-1*np.log10(0.05):
                annotate="n.s."
            else:
                annotate="-log10(p)="+str(p)
        elif significance=="symbol":
            keys=sorted(significance_ranges.keys())
            annotate="NA"
            for j in range(len(keys)):
                if j==0:
                    if p <= significance_ranges[keys[j]]:
                        annotate="n.s."
                        break
                else:
                    if significance_ranges[keys[j-1]] < p <=significance_ranges[keys[j]]:
                        annotate=keys[i]
                        break
            if annotate=="NA":
                annotate=keys[-1]
        plt.text((pval[1]+pval[2])/2, ymax+(ymax-ymin)*(0.06+i*0.051), annotate)
    if significance=="symbol":
        ax.annotate("\n".join(["{}: p < {:.2E}".format(k, 10**(-significance_ranges[k])) for k in keys]),
            xy=(0.9,0.9), xycoords='axes fraction',
            textcoords='offset points',
            size=12,
            bbox=dict(boxstyle="round", fc=(0.9, 0.9, 0.9), ec="none"))
        plt.subplots_adjust(right=0.850)
    
    if yunit!="":
        ax.text(0, 1, "({})".format(yunit), transform=ax.transAxes, ha="right")
    
    if title!="":
        if isinstance(fig, type(None)):
            ax.set_title(title)
        else:
            fig.suptitle(title)

    _save(save, "violin")
    
    return {"p values":newpvals,"axes":ax}



def violinplot2(df: Union[pd.DataFrame, np.ndarray], 
               x: str="", 
               y: str="",
               xlabels: list=[],
               order: list=[],
               pairs: list=[], 
               test: str="ttest_ind",
               alternative: str="two-sided",
               significance: str="numeric",
               significance_ranges: Dict[str, float]={"*":-np.log10(0.05),"**":4,"***":10},
               swarm: bool=False,
               xorder: list=[],
               equal_var: bool=False, 
               yunit: str="",
               title: str="",
               orientation: str="vertical",
               save: str="",
               ax: Optional[plt.Axes]=None,
               scale_prop=False,
               **kwargs):
    """
    Draw a boxplot with a statistical test 
    
    Parameters
    ----------
    df : pandas DataFrame
    
    x,y: str
        names of variables in data
    pairs: list, optional
        Category pairs for the statistical test.
        Examples: [["Adelie","Chinstrap" ],
                    ["Gentoo","Chinstrap" ],
                    ["Adelie","Gentoo" ]]
    test: str, optional
        Method name for the statistical test. Defalt: ttest_ind
        Available methods: ["ttest_ind",
                            "ttest_rel",
                            "kruskal",
                            "mannwhitneyu",
                            "wilcoxon",
                            "brunnermunzel",
                            "median_test"]
    alternative: str ['two-sided', 'less', 'greater'], optional
        Defines the alternative hypothesis. Defalt: "two-sided"
    
    show : bool, optional
        Whether or not to show the figure.
    significance: str ['numeric', 'symbol'], optional
        How to show the significance. 'numeric' will show -log10(p values) in the plot and 
        'symbol' will represent significance as asterisks.
    significance_ranges: dict, optional 
        thresholds of -log10(p values) that each asterisk number represents. Ignored when  significance="numeric".
        example: {"*":-np.log10(0.05),"**":4,"***":10}
    swarm: bool, optional
        Whether or not to superpose a swarm plot. Not recommended if the sample size is too large.
    xorder: list, optional
        The order of x axis labels
    equal_var: bool, optional
        Related to ttest_ind method. The default is True, which will produce a p value equal to t-test in R.
    kwargs: any options accepted by scipy statistical test functions
    
     
    Returns
    -------
    dict("p values":pvalues,"axes":ax)
    
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
    if len(order)!=0:
        xorder=order
    tests=["ttest_ind","ttest_rel","kruskal","mannwhitneyu","wilcoxon","brunnermunzel","median_test"]
    if test not in tests:
        raise ValueError("Available tests are "+", ".join(tests))
    
    if type(df)==pd.DataFrame and len(xorder)==0:
        xorder=sorted(list(set(df[x])))
    
    pvals=[]
    if len(pairs)!=0:
        for p1,p2 in pairs:
            if type(df)==pd.DataFrame:
                p1val, p2val=df[y][df[x]==p1],df[y][df[x]==p2]
            else:
                p1val, p2val=df[xlabels.index(p1)],df[xlabels.index(p2)] 
            statstest=getattr(stats, test)
            if test=="wilcoxon" or test=="ttest_rel":
                _, pval,_=statstest(p1val, p2val,alternative=alternative,**kwargs)
            elif test=="median_test":
                _, pval,_,_=statstest(p1val, p2val,alternative=alternative,**kwargs)
            elif test=="ttest_ind":
                _, pval=statstest(p1val, p2val,alternative=alternative,equal_var=equal_var,**kwargs)
            
            else:
                _, pval=statstest(p1val, p2val,alternative=alternative,**kwargs)
            if len(xlabels)!=0:
                p1ind=xlabels.index(p1)
                p2ind=xlabels.index(p2)
            else:
                p1ind=xorder.index(p1)
                p2ind=xorder.index(p2)
            if pval==0:
                pval=np.inf
            else:
                pval=-np.log10(pval)
            pvals.append([np.abs(p2ind-p1ind), np.amin([p2ind, p1ind]),np.amax([p2ind, p1ind]), pval])
        pvals = sorted(pvals, key = lambda x: (x[0], x[1]))
    if ax==None:
        fig, ax=plt.subplots()
    _violinplot(df=df, x=x,y=y,ax=ax,order=xorder, orientation=orientation, show_points=swarm, scale_prop=scale_prop)
    if type(df)==pd.DataFrame:
        ymax=np.amax(df[y])
    else:
        ymax=np.amax(df)
    newpvals={}
    
    if len(pairs)!=0:
        for i, pval in enumerate(pvals):
            if orientation=="vertical":
                ax.plot([pval[1],pval[2]], [ymax*(1.05+i*0.05),ymax*(1.05+i*0.05)], color="black")
            elif orientation=="horizontal":
                ax.plot([ymax*(1.05+i*0.05),ymax*(1.05+i*0.05)],[pval[1],pval[2]],  color="black")
            p=np.round(pval[-1],2)
            
            newpvals[xorder[pval[1]]+"_"+xorder[pval[2]]]=p
            if significance=="numeric":
                annotate="-log10(p)="+str(p)
            elif significance=="symbol":
                keys=sorted(significance_ranges.keys())
                annotate="NA"
                for j in range(len(keys)):
                    if j==0:
                        if p <= significance_ranges[keys[j]]:
                            annotate="n.s."
                            break
                    else:
                        if significance_ranges[keys[j-1]] < p <=significance_ranges[keys[j]]:
                            annotate=keys[i]
                            break
                if annotate=="NA":
                    annotate=keys[-1]
            if orientation=="vertical":
                ax.text((pval[1]+pval[2])/2, ymax*(1.055+i*0.05), annotate)
            elif orientation=="horizontal":
                ax.text(ymax*(1.055+i*0.05), (pval[1]+pval[2])/2,  annotate, rotation=90)
        if significance=="symbol":
            ax.annotate("\n".join(["{}: p < {:.2E}".format(k, 10**(-significance_ranges[k])) for k in keys]),
                xy=(0.9,0.9), xycoords='axes fraction',
                textcoords='offset points',
                size=12,
                bbox=dict(boxstyle="round", fc=(0.9, 0.9, 0.9), ec="none"))
            plt.subplots_adjust(right=0.850)
    
    if yunit!="":
        if orientation=="vertical":
            ax.text(0, 1, "({})".format(yunit), transform=ax.transAxes, ha="right")
        else:
            ax.text(1, 0, "({})".format(yunit), transform=ax.transAxes, ha="left")
    _save(save, "violin")
    
    return {"p values":newpvals,"axes":ax}

def _violinplot(df: Union[pd.DataFrame, np.ndarray], 
                 x: str="", 
                 y: str="", 
                 xlabels: list=[],
                 order: list=[], 
                 ax: plt.Axes=None,
                 orientation: str="vertical",
                 show_points: bool=True,
                 point_color: str="blue",
                 box_color: str="black",
                 violine_color: Union[str,list, dict]="gray",
                 point_kw={"s":10},
                 scale_prop=False):
    if isinstance(ax, type(None)):
        fig, ax=plt.subplots()
    if isinstance(df, pd.DataFrame):
        if len(order)!=0:
            dfs = {_s: _x[y].values for _s, _x in df.groupby(x)}
            mat=[dfs[_s] for _s in order]
            xlabels=order
        else:
            mat = []
            xlabels=[]
            for _s, _x in df.groupby(x):
                mat.append(_x[y].values)
                xlabels.append(_s)
    else:
        mat=np.array(df)
    
    proportion=np.array([np.sum(mat[i]) for i in range(len(mat))])
    proportion=proportion/np.sum(proportion)
    for i, X in enumerate(mat):
        # X=mat[i]
        # print(X)
        kde=stats.gaussian_kde(X)
        q3, q1=np.quantile(X, 0.75), np.quantile(X, 0.25)
        q2=np.quantile(X, 0.5)
        iqr=q3-q1

        # minx, maxx=q1-iqr*1.5, q3+iqr*1.5
        minx, maxx=np.min(X)-iqr, np.max(X)+iqr
        xinterval=np.linspace(minx,maxx, 100)
        estimate=kde(xinterval)
        if scale_prop is True:
            estimate=proportion[i]*estimate/np.amax(estimate)
        else:
            estimate=0.5*estimate/np.amax(estimate)


        if isinstance(violine_color, list):
            vc=violine_color[i]
        elif isinstance(violine_color, dict):
            vc=xlabels[violine_color[i]]
        elif isinstance(violine_color, str):
            vc=violine_color

        if orientation=="horizontal":
            ax.fill_between(xinterval,i+estimate, i-estimate, color=vc)
            ax.fill_between([q1, q3], [i-0.05,i-0.05], [i+0.05,i+0.05],color=box_color)
            
            ax.plot([q1-iqr*1.5, q3+iqr*1.5], [i,i], lw=1,color=box_color)
            ax.plot([q2, q2], [i-0.05,i+0.05], color="w")
            if show_points is True:
                ax.scatter(X, i+0.5*np.random.uniform(size=X.shape[0])-0.25, color=point_color, **point_kw)
            
        elif orientation=="vertical":
            _yl=np.concatenate([[minx], xinterval, [maxx]])
            _yr=np.concatenate([[minx], xinterval, [maxx]])
            xr=np.concatenate([[i], i+estimate, [i]])
            xl=np.concatenate([[i], i-estimate, [i]])
            ax.fill(xl, _yl, xr,_yr, color=vc )
            #ax.plot([i,i], [q1, q3], lw=10,color=box_color)
            ax.fill([i, i-0.05, i-0.05, i+0.05, i+0.05,i], [q1, q1, q3, q3,q1,q1], lw=10,color=box_color)
            
            ax.plot([i,i], [q1-iqr*1.5, q3+iqr*1.5],  lw=1,color=box_color)
            ax.plot([i-0.05,i+0.05], [q2, q2],  color="w")
            if show_points==True:
                ax.scatter(i+0.5*np.random.uniform(size=X.shape[0])-0.25, X, color=point_color, **point_kw)
    if orientation=="horizontal":
        ax.set_yticks(np.arange(len(mat)), labels=xlabels)
    elif orientation=="vertical":
        ax.set_xticks(np.arange(len(mat)), labels=xlabels, rotation=90)
        plt.subplots_adjust(bottom=0.2)

def _boxplot(df: Union[pd.DataFrame, np.ndarray], 
                 x: str="", 
                 y: str="", 
                 xlabels: list=[],
                 order: list=[], 
                 ax: plt.Axes=None,
                 orientation: str="vertical",
                 show_points: bool=True,
                 point_color: str="blue",
                 box_color: str="black",
                 violine_color: Union[str,list, dict]="gray",
                 point_kw={"s":10},
                 scale_prop=False):
    if isinstance(ax, type(None)):
        fig, ax=plt.subplots()
    if isinstance(df, pd.DataFrame):
        if len(order)!=0:
            dfs = {_s: _x[y].values for _s, _x in df.groupby(x)}
            mat=[dfs[_s] for _s in order]
            xlabels=order
        else:
            mat = []
            xlabels=[]
            for _s, _x in df.groupby(x):
                mat.append(_x[y].values)
                xlabels.append(_s)
    else:
        mat=np.array(df)
    
    proportion=np.array([np.sum(mat[i]) for i in range(len(mat))])
    proportion=proportion/np.sum(proportion)
    for i, X in enumerate(mat):
        # X=mat[i]
        # print(X)
        kde=stats.gaussian_kde(X)
        q3, q1=np.quantile(X, 0.75), np.quantile(X, 0.25)
        q2=np.quantile(X, 0.5)
        iqr=q3-q1

        minx, maxx=q1-iqr*1.5, q3+iqr*1.5
        xinterval=np.linspace(minx,maxx, 100)
        estimate=kde(xinterval)
        if scale_prop is True:
            estimate=proportion[i]*estimate/np.amax(estimate)
        else:
            estimate=0.5*estimate/np.amax(estimate)


        if isinstance(violine_color, list):
            vc=violine_color[i]
        elif isinstance(violine_color, dict):
            vc=xlabels[violine_color[i]]
        elif isinstance(violine_color, str):
            vc=violine_color
        if orientation=="horizontal":
            ax.fill_between(xinterval,i+estimate, i-estimate, color=vc)
            ax.fill_between([q1, q3], [i-0.05,i-0.05], [i+0.05,i+0.05],color=box_color)
            
            ax.plot([q1-iqr*1.5, q3+iqr*1.5], [i,i], lw=1,color=box_color)
            ax.plot([q2, q2], [i-0.05,i+0.05], color="w")
            if show_points is True:
                ax.scatter(X, i+0.5*np.random.uniform(size=X.shape[0])-0.25, color=point_color, **point_kw)
            
        elif orientation=="vertical":
            _yl=np.concatenate([[minx], xinterval, [maxx]])
            _yr=np.concatenate([[minx], xinterval, [maxx]])
            xr=np.concatenate([[i], i+estimate, [i]])
            xl=np.concatenate([[i], i-estimate, [i]])
            ax.fill(xl, _yl, xr,_yr, color=vc )
            #ax.plot([i,i], [q1, q3], lw=10,color=box_color)
            ax.fill([i, i-0.05, i-0.05, i+0.05, i+0.05,i], [q1, q1, q3, q3,q1,q1], lw=10,color=box_color)
            
            ax.plot([i,i], [q1-iqr*1.5, q3+iqr*1.5],  lw=1,color=box_color)
            ax.plot([i-0.05,i+0.05], [q2, q2],  color="w")
            if show_points is True:
                ax.scatter(i+0.5*np.random.uniform(size=X.shape[0])-0.25, X, color=point_color, **point_kw)
    if orientation=="horizontal":
        ax.set_yticks(np.arange(len(mat)), labels=xlabels)
    elif orientation=="vertical":
        ax.set_xticks(np.arange(len(mat)), labels=xlabels, rotation=90)
        plt.subplots_adjust(bottom=0.2)

def lineplot(df: pd.DataFrame,
             x: str="", 
             y: Union[str, list]="", 
             variables: Union[str, list]="",
             split: bool=False,
             ax: plt.Axes=None,
             palette: Union[str, dict]="tab20b",
             xlabel: str="",
             ylabel: str="",
             xunit: str="",
             yunit: Union[str, dict]="",
             xformat: str="",
             yformat: str="",
             logscalex: bool=False,
             logscaley: bool=False,
             title: str="",
             plotkw: dict={},
             show_legend: bool=True,
             bbox_to_anchor: Union[list, tuple]=[1.05, 1],
             right=0.75,
             left=0.15,
             bottom=0.15,
             figsize: list=[],
             rows_cols: list=[],
             estimate: str="",
             robust_param: dict={},
             linestyle: Union[bool, dict, list, str]=False) ->Dict:
    """
    Drawing line plots with some extra features.
    
    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to be plotted
    x: str, optional (default="")
        Column name of x-axis
    y: Union[str, list], optional (default="")
        Column name of y-axis
    variables: Union[str, list], optional (default="")
        Column name of variables
    split: bool, optional (default=False)
        If True, split the dataframe by variables
    ax: plt.Axes, optional (default=None)
        Axes to be plotted
    palette: Union[str, dict], optional (default="tab20b")
        Color palette
    xlabel: str, optional (default="")
        Label of x-axis
    ylabel: str, optional (default="")
        Label of y-axis
    xunit: str, optional (default="")
        Unit of x-axis
    yunit: Union[str, dict], optional (default="")
        Unit of y-axis
    xformat: str, optional (default="")
        Format of x-axis
    yformat: str, optional (default="")
        Format of y-axis
    logscalex: bool
        If True, x-axis is in log scale
    logscaley: bool
        If True, y-axis is in log scale
    title: str
        Title of the plot
    plotkw: dict
        Keyword arguments for plt.plot
    show_legend: bool
        If True, show legend
    bbox_to_anchor: Union[list, tuple]
        Bbox_to_anchor for legend
    right: float
        Right margin
    left: float
        Left margin
    bottom: float
        Bottom margin
    figsize: list
        Figure size
    rows_cols: list
        List of rows and columns for subplots
    estimate: str="", optional (default=""), ["moving_average", "moving_median", "regression"]
        Estimate method for lineplot
    robust_param: dict={}
        Parameters for robust regression
    
    Returns
    -------
        {"axes":ax, "stats":stats}: dict
    """
    if len(y)==0 and len(variables)==0:
        raise ValueError("Provide y or variables option")
    elif len(variables)!=0:
        y=copy.deepcopy(variables)

    if isinstance(y, str):
        y=[copy.deepcopy(y)]

    if len(x)==0:
        X=list(df.index)
        x=df.index.name
    else:
        X=df[x]

    
    if isinstance(palette, str):
        cmap=plt.get_cmap(palette, len(y))
        lut={}
        for i, _y in enumerate(y):
            lut[_y]=cmap(i)
    else:
        lut=copy.deepcopy(palette)

    line_lut={}
    if isinstance(linestyle, str):
        for _y in y:
            line_lut[_y]=linestyle
    elif isinstance(linestyle, bool):
        if linestyle is True:
            for _l, _y in zip(linestyle_list, y):
                line_lut[_y]=_l
        else:
            for _y in y:
                line_lut[_y]="-"
    elif isinstance(linestyle, dict):
        line_lut=linestyle
    elif isinstance(linestyle, list):
        for _l, _y in zip(linestyle_list, y):
            line_lut[_y]=_l

    _stats={}

    if split==False:
        if len(figsize)==0:
            figsize=[6,4]
        if isinstance(ax, type(None)):
            fig, ax=plt.subplots(figsize=figsize)
        for _y in y:
            ax.plot(X, df[_y], color=lut[_y], label=_y, linestyle=line_lut[_y], **plotkw)
            if estimate=="moving_average":
                ax.plot(X, df[_y].rolling(window=int(X.shape[0]/10), center=True).mean(), color="black", linestyle="dashed")
            elif estimate=="moving_median":
                ax.plot(X, df[_y].rolling(window=int(X.shape[0]/10), center=True).median(), color="black", linestyle="dashed")
            elif estimate=="regression":
                print(X.min(), X.max())
                plotline_X = np.arange(X.min(), X.max()).reshape(-1, 1)
                (fitted_model, summary, 
                 coef, coef_p, intercept, intercept_p, r2, x_line, y_line, ci, pi,std_error, 
                 MSE)=_robust_regression(X, df[_y].fillna(0), plotline_X, robust_param)
                _stats[_y]={"model": fitted_model, "summary": summary, "coef": coef, 
                           "coef_pval": coef_p, "intercept": intercept, 
                           "intercept_pval": intercept_p, "r2": r2, "x_line": x_line, "y_line": y_line, 
                           "ci": ci, "pi": pi, "std_error": std_error, "MSE": MSE}
                ax.plot(x_line, y_line, color="black", linestyle="dashed")
                _draw_ci_pi(ax, ci, pi,x_line, y_line, label=False)
        _set_axis(ax,x, xlabel, ylabel, xunit, yunit, xformat, yformat,logscalex,logscaley, title)

        if show_legend is True:
            plt.legend(bbox_to_anchor=bbox_to_anchor)
            plt.subplots_adjust(right=right, bottom=bottom, left=left)
    else:
        plotnum=len(y)
        if len(rows_cols)==0:
            rows_cols=[plotnum//3+int(plotnum%3!=0), 3]
        if len(figsize)==0:
            figsize=[6,2*(plotnum//3+int(plotnum%3!=0))]
        if isinstance(ax, type(None)):
            fig, ax=plt.subplots(nrows=rows_cols[0], ncols=rows_cols[1], figsize=figsize)
            ax=ax.flatten()
        
        for _ax, _y in zip(ax, y):
            _ax.plot(X, df[_y], color=lut[_y], linestyle=line_lut[_y], **plotkw)
            if estimate=="moving_average":
                _ax.plot(X, df[_y].rolling(window=int(X.shape[0]/10), center=True).mean(), color="black", linestyle="dashed")
            elif estimate=="moving_median":
                _ax.plot(X, df[_y].rolling(window=int(X.shape[0]/10), center=True).median(), color="black", linestyle="dashed")
            elif estimate=="regression":
                # print(X.min(), X.max())
                plotline_X = np.arange(X.min(), X.max()).reshape(-1, 1)
                (fitted_model, summary, 
                 coef, coef_p, intercept, intercept_p, r2, x_line, y_line, ci, pi,std_error, 
                 MSE)=_robust_regression(X, df[_y].fillna(0), plotline_X, robust_param)
                _stats[_y]={"model": fitted_model, "summary": summary, "coef": coef, 
                           "coef_pval": coef_p, "intercept": intercept, 
                           "intercept_pval": intercept_p, "r2": r2, "x_line": x_line, "y_line": y_line, 
                           "ci": ci, "pi": pi, "std_error": std_error, "MSE": MSE}
                _ax.plot(x_line, y_line, color="black", linestyle="dashed")
                _draw_ci_pi(_ax, ci, pi,x_line, y_line)
                # print(coef_p)
                if np.isnan(coef_p):
                    mlog10=np.nan
                elif coef_p<=10**(-320):
                    mlog10=np.inf
                elif coef_p>10**(-320):
                    mlog10=-1*np.log10(coef_p)
                _ax.text(0, 0.8, "coef: {:.3f} (-log10p: {:.1f})\nR2: {:.3f}".format(coef, mlog10, r2), transform=_ax.transAxes, ha="left", fontsize=8)
            _set_axis(_ax,x, xlabel, _y, xunit, yunit, xformat, yformat,logscalex,logscaley, "")
        fig.suptitle(title)
        plt.tight_layout(h_pad=0, w_pad=0)
    return {"axes":ax, "stats":_stats}

def _set_axis(ax,x, xlabel, ylabel, xunit, yunit, xformat, yformat,logscalex,logscaley, title):
    if xlabel !="":
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(x)
    if ylabel!="":
        ax.set_ylabel(ylabel)
    ax.set_title(title)
    if yunit!="":
        ax.text(0, 1, "({})".format(yunit), transform=ax.transAxes, ha="right")
    if xunit!="":
        ax.text(1, 0, "({})".format(xunit), transform=ax.transAxes, ha="left",va="top")
    if xformat!="":
        
        ax.xaxis.set_major_formatter(StrMethodFormatter(xformat))
    if yformat !="":
        ax.yaxis.set_major_formatter(StrMethodFormatter(yformat))
    if logscalex==True:
        ax.set_xscale("log")
    if logscaley==True:
        ax.set_yscale("log")
    ax.tick_params(pad=1,axis='both', which='both', length=0)
if __name__=="__main__":
    pass