from typing import Union, Optional, Dict, List
import matplotlib.collections as mc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import cm
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import leaves_list
from scipy.cluster import hierarchy
from collections import defaultdict
import matplotlib.colors
from natsort import natsort_keygen, natsorted
from matplotlib.patches import Rectangle
import scipy.cluster.hierarchy as sch
import fastcluster as fcl

import sys 
import matplotlib as mpl
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA, NMF, LatentDirichletAllocation
from scipy.stats import fisher_exact
from scipy.stats import zscore
from itertools import combinations
import os
#script_dir = os.path.dirname( __file__ )
#sys.path.append( script_dir )
from omniplot.utils import * #
import scipy.stats as stats
from joblib import Parallel, delayed
from omniplot.chipseq_utils import _calc_pearson
import itertools as it
from omniplot.scatter import *
from omniplot.proportion import *
from omniplot.heatmap import *
from omniplot.utils import colormap_list, hatch_list, marker_list
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
               y: list=[],linewidth: float=1,figsize: Optional[list]=None,
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
    # if len(variables)!=0 and len(category)!=0:
    #     if type(category)==str:
    #         category=[category]
    #     category_df=df[category]
    #     df=df[variables]
    #     X = df.values
    #     #print(X)
    #     assert X.dtype==float, f"{x} columns must contain only float values."
    #
    #
    # elif len(category) !=0:
    #     if type(category)==str:
    #         category=[category]
    #     category_df=df[category]
    #     df=df.drop(category, axis=1)
    #     X = df.values
    #     #print(X)
    #     assert X.dtype==float, f"data must contain only float values except {category} column."
    #
    # else:    
    #     X = df.values
    #     assert X.dtype==float, "data must contain only float values."
    
    
    
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


def violinplot(df: pd.DataFrame, 
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
               ax: Optional[plt.Axes]=None,**kwargs):
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
    tests=["ttest_ind","ttest_rel","kruskal","mannwhitneyu","wilcoxon","brunnermunzel","median_test"]
    if not test in tests:
        raise Exception("Available tests are "+", ".join(tests))
    import scipy.stats as stats
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
        
    fig, ax=plt.subplots()
    sns.violinplot(data=df, x=x,y=y,inner="quartile")
    if swarm==True:
        sns.swarmplot(data=df, x=x,y=y,color="black",alpha=0.5)
    ymax=np.amax(df[y])
    newpvals={}
    for i, pval in enumerate(pvals):
        plt.plot([pval[1],pval[2]], [ymax*(1.05+i*0.05),ymax*(1.05+i*0.05)], color="black")
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
                        annotate=""
                        break
                else:
                    if significance_ranges[keys[j-1]] < p <=significance_ranges[keys[j]]:
                        annotate=keys[i]
                        break
            if annotate=="NA":
                annotate=keys[-1]
        plt.text((pval[1]+pval[2])/2, ymax*(1.055+i*0.05), annotate)
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
    if not test in tests:
        raise Exception("Available tests are "+", ".join(tests))
    import scipy.stats as stats
    if type(df)==pd.DataFrame and len(xorder)==0:
        xorder=sorted(list(set(df[x])))
    
    pvals=[]
    if len(pairs)!=0:
        for p1,p2 in pairs:
            if type(df)==pd.DataFrame:
                p1val, p2val=df[y][df[x]==p1],df[y][df[x]==p2]
            else:
                p1val, p2val=mat[xlabels.index(p1)],mat[xlabels.index(p2)] 
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
    _violinplot(df=df, x=x,y=y,ax=ax, orientation=orientation, show_points=swarm, scale_prop=scale_prop)
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
                            annotate=""
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
    if ax==None:
        fig, ax=plt.subplots()
    if type(df)==pd.DataFrame:
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
    for i in range(len(mat)):
        X=mat[i]
        # print(X)
        kde=stats.gaussian_kde(X)
        q3, q1=np.quantile(X, 0.75), np.quantile(X, 0.25)
        q2=np.quantile(X, 0.5)
        iqr=q3-q1

        minx, maxx=q1-iqr*1.5, q3+iqr*1.5
        xinterval=np.linspace(minx,maxx, 100)
        estimate=kde(xinterval)
        if scale_prop==True:
            estimate=proportion[i]*estimate/np.amax(estimate)
        else:
            estimate=0.5*estimate/np.amax(estimate)


        if type(violine_color)==list:
            vc=violine_color[i]
        elif type(violine_color)==dict:
            vc=xlabels[violine_color[i]]
        elif type(violine_color)==str:
            vc=violine_color
        if orientation=="horizontal":
            ax.fill_between(xinterval,i+estimate, i-estimate, color=vc)
            ax.fill_between([q1, q3], [i-0.05,i-0.05], [i+0.05,i+0.05],color=box_color)
            
            ax.plot([q1-iqr*1.5, q3+iqr*1.5], [i,i], lw=1,color=box_color)
            ax.plot([q2, q2], [i-0.05,i+0.05], color="w")
            if show_points==True:
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
    if ax==None:
        fig, ax=plt.subplots()
    if type(df)==pd.DataFrame:
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
    for i in range(len(mat)):
        X=mat[i]
        # print(X)
        kde=stats.gaussian_kde(X)
        q3, q1=np.quantile(X, 0.75), np.quantile(X, 0.25)
        q2=np.quantile(X, 0.5)
        iqr=q3-q1

        minx, maxx=q1-iqr*1.5, q3+iqr*1.5
        xinterval=np.linspace(minx,maxx, 100)
        estimate=kde(xinterval)
        if scale_prop==True:
            estimate=proportion[i]*estimate/np.amax(estimate)
        else:
            estimate=0.5*estimate/np.amax(estimate)


        if type(violine_color)==list:
            vc=violine_color[i]
        elif type(violine_color)==dict:
            vc=xlabels[violine_color[i]]
        elif type(violine_color)==str:
            vc=violine_color
        if orientation=="horizontal":
            ax.fill_between(xinterval,i+estimate, i-estimate, color=vc)
            ax.fill_between([q1, q3], [i-0.05,i-0.05], [i+0.05,i+0.05],color=box_color)
            
            ax.plot([q1-iqr*1.5, q3+iqr*1.5], [i,i], lw=1,color=box_color)
            ax.plot([q2, q2], [i-0.05,i+0.05], color="w")
            if show_points==True:
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


if __name__=="__main__":
    pass