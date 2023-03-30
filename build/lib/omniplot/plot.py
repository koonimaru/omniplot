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
colormap_list: list=["nipy_spectral", 
                     "terrain",
                     "tab20b",
                     "tab20c",
                     "gist_rainbow",
                     "hsv",
                     "CMRmap",
                     "coolwarm",
                     "gnuplot",
                     "gist_stern",
                     "brg",
                     "rainbow",
                     "jet"]
hatch_list: list = ['//', '\\\\', '||', '--', '++', 'xx', 'oo', 'OO', '..', '**','/o', '\\|', '|*', '-\\', '+o', 'x*', 'o-', 'O|', 'O.', '*-']
maker_list: list=['.', '_' , '+','|', 'x', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'D', 'd', 'P', 'X','o', '1', '2', '3', '4','|', '_']




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


def correlation(df: pd.DataFrame, 
                category: Union[str, list]=[],
                variables: List=[],
                method="pearson",
                palette: str="coolwarm",
                figsize=[6,6],
                show_values=False,
                clustermap_param:dict={},
                ztransform: bool=True,
                xticklabels =False,
                yticklabels=False,
                title: str="",)->Dict:
    """
    Drawing a heatmap with correlations or distances between observations 
    
    Parameters
    ----------
    df : pandas DataFrame
        
    variables: List, optional
        the names of values to calculate correlations  
    
    category: str or list, optional
        the names of categorical values to display as color labels
    mthod: str
        method for correlation/distance calculation. Defalt: "pearson"
        
    palette : str
        A colormap name
    show_values: bool, optional
        Wheter to exhibit the values of fractions/counts/percentages.
    
    clustermap_param : dict, optional
        Whether or not to show the figure.
    
    figsize : List[int], optional
        The figure size, e.g., [4, 6].
    ztransform : bool, optional
        Whether to transform values to z-score
    xticklabels, yticklabels : bool
        Whether to show the label names in the heatmap
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
    original_index=df.index
    X, category=_separate_data(df, variables=variables, category=category)
    # if len(category) !=0:
    #
    #     if type(category)==str:
    #         category=[category]
    #     #df=df.drop(category, axis=1)
    #     valnames=list(set(df.columns) -set(category)) 
    #     X = df[valnames].values
    #     assert X.dtype==float, f"data must contain only float values except {category} column."
    #
    # else:    
    #     X = df.values
    #     assert X.dtype==float, "data must contain only float values."
    if ztransform==True:
        X=zscore(X, axis=0)
    if method=="pearson":
        # dmat=Parallel(n_jobs=-1)(delayed(_calc_pearson)(ind, X) for ind in list(it.combinations(range(X.shape[0]), 2)))
        # dmat=np.array(dmat)
        # dmat=squareform(dmat)
        # print(dmat)
        # dmat+=np.identity(dmat.shape[0])
        dmat=np.corrcoef(X)
    else:
        dmat=squareform(pdist(X, method))
    if method=="pearson":
            ctitle="Pearson correlation"
    else:
        ctitle=method+" distance"    
        
        
    if len(category) >0:
        dfm=pd.DataFrame(data=dmat)
        colnames=dfm.columns
        for cat in category:
            dfm[cat]=df[cat].values
        res=complex_clustermap(dfm,
                               heatmap_col=colnames, 
                               row_colors=category,
                               ztranform=False,
                               xticklabels=xticklabels,
                               yticklabels=yticklabels,
                               figsize=figsize,
                               ctitle=ctitle )
    else:
        
        res=complex_clustermap(data=dmat,
                         xticklabels=xticklabels,
                         yticklabels=yticklabels,
                   method="ward", 
                   cmap=palette,
                   col_cluster=True,
                   row_cluster=True,
                   figsize=figsize,
                   #rasterized=True,
                    #cbar_kws={"label":"Pearson correlation"}, 
                   #annot=show_values,
                   **clustermap_param)
    return res
        # g.cax.set_ylabel(ctitle, rotation=-90,va="bottom")
        # plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)  # For y axis
        # plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90) # For x axis
        # return {"grid":g}

def triangle_heatmap(df, 
                     grid_pos: list=[],
                     grid_labels: list=[],
                     show: bool=False, 
                     save: str="",title: str="")-> dict:
    
    """
    Creating a heatmap with 45 degree rotation.
    
    Parameters
    ----------
    df : pandas DataFrame
    grid_pos: list
        the column name of a category that is going to be placed in the row of the dotplot
    grid_labels: list
        the column name of a category that is going to be placed in the column of the dotplot
    show : bool
        Whether or not to show the figure.
    
    Returns
    -------
    dict {"axes": ax}
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
    
    
    genes=df.index
    fig, ax = plt.subplots(figsize=[8,6])
    dmat=df.to_numpy()
    D = dmat
    N = dmat.shape[0]
    a=np.tril(np.zeros([N,N])-1000000, k=-1)
    # Get the lower triangle of the matrix. 
    C = np.triu(D)+a
    
    # Mask the upper triangle.
    C = np.ma.masked_array(C, C == -1000000)
    # Set the diagonal to zero.
    for i in range(N):
        C[i, i] = 0
    
    # Transformation matrix for rotating the heatmap.
    A = np.array([(y, x) for x in range(N, -1, -1) for y in range(N + 1)])
    t = np.array([[2**(-0.5), 2**(-0.5)], [-2**(-0.5), 2**(-0.5)]])
    A = np.dot(A, t)
    #t_ = np.array([[2**(-0.5), -2**(-0.5)], [2**(-0.5), 2**(-0.5)]])
    
    # -1.0 correlation is blue, 0.0 is white, 1.0 is red.
    cmap = plt.cm.Reds
    #norm = mp.colors.BoundaryNorm(np.linspace(0, 10, 14), cmap.N)
    
    # This MUST be before the call to pl.pcolormesh() to align properly.
    ax.set_xticks([])
    ax.set_yticks([])
    
    X = A[:, 1].reshape(N + 1, N + 1)
    Y = A[:, 0].reshape(N + 1, N + 1)
    caxes = plt.pcolormesh(X, Y, np.flipud(C), axes=ax, cmap=cmap, rasterized=True)
    x1s=[0]
    if len(grid_pos)>0:
        for i,  grid in enumerate(grid_pos):
            x0=(grid)/(2**(0.5))
            x1=(grid)*(2**0.5)
            y0=(grid)/(2**(0.5))
            y1=0
            ax.plot([x0, x1], [y0, y1],color='gray', linewidth=1)
            ax.plot([(grid)*(2**0.5), (grid)*(2**0.5)+(N-grid)/(2**(0.5))], [0, (N-grid)/(2**(0.5))], color='gray', linewidth=1)
            x1s.append(x1)
    x1s.append(N*2**0.5)
    
    

    if len(genes) >0: 
        leng=0
        for i, g in enumerate(genes):
            #ax.plot([(i+0.5)*(2**0.5), (i+0.5)*(2**0.5)], [-2**0.5, -(N/10)*2**0.5/2],color='b', linewidth=0.5)
            ax.text((i+0.5)*(2**0.5), -(N/10)*2**0.5/2*1.01, g, rotation=90,ha='center', va='top', fontsize="small")
            leng+=len(g)
        leng=leng/len(genes)
    else:
        leng=0
    spacing=(N/10)*leng*0.4
    
    
    rect=Rectangle([0, -spacing],N*(2**0.5), spacing, color='whitesmoke' , alpha=1, linewidth=0)
    ax.add_patch(rect)
    
    
    rect=Rectangle([0, -(N/10)*2**0.5/2],N*(2**0.5), (N/10)*2**0.5/2, color='dimgray' , alpha=1, linewidth=0)
    ax.add_patch(rect)
    
    if len(grid_labels)>0:
        for i in range(len(x1s)-1):
            if i%2==1:
                rect=Rectangle([x1s[i], -spacing],np.abs(x1s[i] - x1s[i+1]), spacing, color='silver' , alpha=0.3, linewidth=0)
                ax.add_patch(rect)
            else:
                rect=Rectangle([x1s[i], -spacing],np.abs(x1s[i] - x1s[i+1]), spacing, color='lavender' , alpha=0.3, linewidth=0)
                ax.add_patch(rect)
            x=(x1s[i]+x1s[i+1])/2
            ax.text(x,-(N/10)*2**0.5/4, grid_labels[i], rotation=90,ha='center', va='center', color="w")
    

    
    
    cb = plt.colorbar(caxes, ax=ax, shrink=0.75)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    if save!="":
        if save.endswith(".pdf") or save.endswith(".png") or save.endswith(".svg"):
            plt.savefig(save)
        else:
            plt.savefig(save+"_complexheatmap.pdf")
    if show:
        plt.show()
    return {"axes": ax}
    
    
def complex_clustermap(df: pd.DataFrame,
                       variables: list=[],
                       dfcol: Optional[pd.DataFrame]=None, 
                       row_colors: list=[],
                       col_colors: list=[],
                       row_plot: list=[],
                       col_plot: list=[],
                       row_scatter: list=[],
                       col_scatter: list=[],
                       row_bar: list=[],
                       col_bar: list=[],
                       ctitle: str="",
                       approx_clusternum: int=10,
                       approx_clusternum_col: int=3,
                       color_var: int=0,
                       merginalsum: bool=False,
                       show: bool=False,
                       method: str="ward",
                       return_col_cluster: bool=True,
                       ztranform: bool=True,
                       xticklabels: bool=True, 
                       yticklabels: bool=False,
                       show_plot_labels: bool=False,
                       figsize: list=[],
                       title: str="",
                       save: str="",
                       heatmap_palette: str="coolwarm",
                       heatmap_col: list=[],
                       **kwargs):
    """
    Drawing a clustered heatmap with merginal plots.
    
    Parameters
    ----------
    df : pandas DataFrame
    variables : list
        the column names of variables for heatmap
    row_colors, col_colors: list, optional
        the column names of categorical values to be plotted as color labels. for the col_colors to be plotted, dfcol options will be needed.
    row_plot, col_plot : list, optional
        The column names for the values to be plotted as lines.
    row_scatter, col_scatter: list, optional
        The column names for the values to be plotted as points.
    row_bar, col_bar: list, optional
        The column names for the values to be plotted as bars.
    approx_clusternum : int, optional
        The approximate number of row clusters to be created. Labeling the groups of leaves with different colors. The result of hierarchical clustering won't change.    
    approx_clusternum_col : int, optional
        The approximate number of column clusters to be created. Labeling the groups of leaves with different colors. The result of hierarchical clustering won't change.

    color_var : int, optional
        The number of potential colors in dendrograms. If some clusters in the dendrogram share a same color (because the number of clusters is too many), 
        give this option may solve the problem. 
    merginalsum : bool, optional
        Whether or not to draw bar plots for merginal distribution.
    show : bool, optional
        Whether or not to show the figure.
    method : string, optional (default: "ward")
        Method for hierarchical clustering. ["ward", "single", 
    return_col_cluster : string
        The title for color values. If not set, "color_val" will be used.
    
    ztranform: bool, optional (default: True)
    xticklabels: bool, optional (default: True)
        Whether or not to show xtick labels
    yticklabels: bool, optional (default: False)
        Whether or not to show ytick labels
    show_plot_labels: bool, optional (default: False)
        Whether or not to show plot labels.
    figsize: list, optional
        
    title: str, optional
    save: str, optional
    
    heatmap_col: list, optional
        The same as the "variables" option. Will be deprecated.
    Returns
    -------
        {"row_clusters":pd.DataFrame,"col_clusters":pd.DataFrame, "grid":g}: dict
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
    """#print(kwargs)
    rnum, cnum=df.shape
    if len(heatmap_col)==0 and len(variables)==0:
        raise Exception("Please specify the variables option")
    if len(heatmap_col)!=0:
        variables=heatmap_col
    cnum=len(variables)
    if len(figsize)==0:
        xsize=np.amin([cnum, 20])
        xsize=np.amax([xsize, 5])
        figsize=[xsize,10]
    scatterpointsize=5
    sns.set(font_scale=1)
    if ztranform==True:
        df[variables]=df[variables].apply(zscore)
        if ctitle =="":
            ctitle="zscore"
    
    if len(col_plot)!=0 or len(col_scatter)!=0 or len(col_bar)!=0:
        if dfcol==None:
            raise Exception("if you want to plot along the x axis, you need provide dfcol option containing values to plot.")
    cdata={"Cluster":[],"Index":[],"RGB":[]}
    totalrowplot=0
    if merginalsum==True:
        totalrowplot+=1
    totalrowplot+=len(row_plot)
    totalrowplot+=len(row_colors)
    totalrowplot+=len(row_scatter)
    totalrowplot+=len(row_bar)
    totalcolplot=0
    if merginalsum==True:
        totalcolplot+=1
    totalcolplot+=len(col_plot) 
    totalcolplot+=len(col_colors)
    totalcolplot+=len(col_scatter)
    totalcolplot+=len(col_bar)
    _row_color_legend={}
    _col_color_legend={}
    colormap_index=0
    if totalrowplot + totalcolplot >0:
        rowplotcount=0
        colplotcount=0
        _row_colors=[]
        _row_colors_title=[]
        _col_colors=[]
        _col_colors_title=[]
        
        if merginalsum:
            _row_colors.append(np.ones([rnum, 4]))
            _row_colors_title.append("Sum")
            _col_colors.append(np.ones([cnum, 4]))
            _col_colors_title.append("Sum")
        #print(np.shape(_col_colors))
        if len(row_colors)>0:
            for k in row_colors:
                
                u=np.array(natsorted(np.unique(df[k])))
                _cmap=plt.get_cmap(colormap_list[colormap_index],u.shape[0])
                lut={}
                for _i, _u in enumerate(u):
                    lut[_u]=_cmap(_i)
                _row_color_legend[k]=lut
                colormap_index+=1
                _row_colors.append([lut[cat] for cat in df[k]])
                _row_colors_title.append(k)
                
        if len(col_colors)>0:
            for k in col_colors:
                u=np.array(natsorted(np.unique(dfcol[k])))
                _cmap=plt.get_cmap(colormap_list[colormap_index],u.shape[0])
                lut={}
                for _i, _u in enumerate(u):
                    lut[_u]=_cmap(_i)
                _col_color_legend[k]=lut
                colormap_index+=1
                _col_colors.append([lut[cat] for cat in dfcol[k]])
                _col_colors_title.append(k)
        
        if len(row_plot)>0:
            for k in row_plot:
                _row_colors.append(np.ones([rnum, 4]))
                _row_colors_title.append(k)
        if len(col_plot)>0:
            for k in col_plot:
                _col_colors.append(np.ones([rnum, 4]))
                _col_colors_title.append(k)        
        
        if len(row_scatter)>0:
            for k in row_scatter:
                _row_colors.append(np.ones([rnum, 4]))
                _row_colors_title.append(k)
        if len(col_scatter)>0:
            for k in col_scatter:
                _col_colors.append(np.ones([rnum, 4]))
                _col_colors_title.append(k) 
        
        if len(row_bar)>0:
            for k in row_bar:
                _row_colors.append(np.ones([rnum, 4]))
                _row_colors_title.append(k)
        if len(col_bar)>0:
            for k in col_bar:
                _col_colors.append(np.ones([rnum, 4]))
                _col_colors_title.append(k) 
        
        if len(_row_colors) >0 and len(_col_colors) >0:
            #print(np.shape(_col_colors))
            g=sns.clustermap(df[variables],col_colors=_col_colors, 
                             row_colors=_row_colors,
                             method=method,xticklabels=xticklabels, yticklabels=yticklabels,
                             figsize=figsize,dendrogram_ratio=0.1,
                             cbar_kws=dict(cmap=heatmap_palette, orientation='horizontal'),
                             **kwargs)
            
            g.ax_col_colors.invert_yaxis()
            g.ax_row_colors.invert_xaxis()
        elif len(_col_colors) >0:
           
            g=sns.clustermap(df[variables],
                             col_colors=_col_colors,
                             method=method,
                             xticklabels=xticklabels, 
                             yticklabels=yticklabels,
                             dendrogram_ratio=0.1,
                             cbar_kws=dict(cmap=heatmap_palette, orientation='horizontal'),
                             figsize=figsize,**kwargs)
            g.ax_col_colors.invert_yaxis()
        elif len(_row_colors) >0:
            g=sns.clustermap(df[variables],
                             row_colors=_row_colors,
                             method=method,
                             xticklabels=xticklabels,
                             cmap=heatmap_palette, 
                             yticklabels=yticklabels,
                             dendrogram_ratio=0.1,
                             figsize=figsize,
                             cbar_kws=dict(orientation='horizontal'),
                             **kwargs)
            g.ax_row_colors.invert_xaxis()
        
        
        #for spine in g.ax_cbar.spines:
        #    g.ax_cbar.spines[spine].set_linewidth(2)

        
        rowplotcount=0
        colplotcount=0
        tickpos=0.9
        row_labels=[]
        col_labels=[]
        row_ticks=[]
        col_ticks=[]
        if merginalsum:
            mat=df[variables].to_numpy()
            r=np.sum(mat, axis=1)
            row_labels.append(0)
            row_labels.append(np.amax(r))
            row_ticks.append(rowplotcount)
            row_ticks.append(rowplotcount+tickpos)
            row_cluster=True
            if "row_cluster" in kwargs:
                row_cluster=kwargs["row_cluster"]
            if row_cluster==True:
                g.ax_row_colors.barh(np.arange(r.shape[0])+0.5, r[leaves_list(g.dendrogram_row.linkage)]/np.amax(r),height=1)
                
            else:
                g.ax_row_colors.barh(np.arange(r.shape[0])+0.5, r/np.amax(r),height=1)
            #g.ax_row_colors.set_xticks([0,1],labels=[0,np.amax(r)])
            c=np.sum(mat, axis=0)
            #print(mat, c)
            col_cluster=True
            if "col_cluster" in kwargs:
                col_cluster=kwargs["col_cluster"]
            if col_cluster==True:
            
            #print(leaves_list(g.dendrogram_col.linkage))
                g.ax_col_colors.bar(np.arange(c.shape[0])+0.5,c[leaves_list(g.dendrogram_col.linkage)]/np.amax(c),width=1)
            else:
                g.ax_col_colors.bar(np.arange(c.shape[0])+0.5,c/np.amax(c),width=1)
            #g.ax_col_colors.set_yticks([0,1],labels=[0,np.amax(c)])
            col_labels.append(0)
            col_labels.append(np.amax(c))
            col_ticks.append(colplotcount)
            col_ticks.append(colplotcount+tickpos)
            rowplotcount=1
            colplotcount=1
        rowplotcount+=len(row_colors)
        colplotcount+=len(col_colors)
        if len(row_plot)>0:
            row_cluster=True
            if "row_cluster" in kwargs:
                row_cluster=kwargs["row_cluster"]
            
            for i, lname in enumerate(row_plot):
                r=np.array(df[lname])
                row_labels.append(np.amin(r))
                row_labels.append(np.amax(r))
                row_ticks.append(rowplotcount)
                row_ticks.append(rowplotcount+tickpos)

                r=r-np.amin(r)
                r=r/np.amax(r)
                r=0.9*r
                if row_cluster==True:
                    tmpindx=leaves_list(g.dendrogram_row.linkage)
                    r=r[tmpindx]
                    
                    g.ax_row_colors.plot(r+rowplotcount, np.arange(r.shape[0])+0.5)
                else:
                    g.ax_row_colors.plot(r+rowplotcount, np.arange(r.shape[0])+0.5)
            
                rowplotcount+=1
                
        if len(col_plot)>0:
            col_cluster=True
            if "col_cluster" in kwargs:
                col_cluster=kwargs["col_cluster"]
            for i, lname in enumerate(col_plot):
                r=np.array(dfcol[lname])
                col_labels.append(np.amin(r))
                col_labels.append(np.amax(r))
                col_ticks.append(colplotcount)
                col_ticks.append(colplotcount+tickpos)
                
                r=r-np.amin(r)
                r=r/np.amax(r)
                r=0.9*r
                if col_cluster==True:
                    g.ax_col_colors.plot(np.arange(r.shape[0])+0.5,r[leaves_list(g.dendrogram_col.linkage)]+colplotcount)
                else:
                    g.ax_col_colors.plot(np.arange(r.shape[0])+0.5,r+colplotcount)
                
                colplotcount+=1
        if len(row_scatter)>0:
            row_cluster=True
            if "row_cluster" in kwargs:
                row_cluster=kwargs["row_cluster"]
            
            for i, lname in enumerate(row_scatter):
                r=np.array(df[lname])
                row_labels.append(np.amin(r))
                row_labels.append(np.amax(r))
                row_ticks.append(rowplotcount)
                row_ticks.append(rowplotcount+tickpos)
                
                r=r-np.amin(r)
                r=r/np.amax(r)
                r=0.9*r
                if row_cluster==True:
                    tmpindx=leaves_list(g.dendrogram_row.linkage)
                    r=r[tmpindx]
                    g.ax_row_colors.scatter(r+rowplotcount, np.arange(r.shape[0])+0.5,s=scatterpointsize)
                else:
                    g.ax_row_colors.scatter(r+rowplotcount, np.arange(r.shape[0])+0.5,s=scatterpointsize)
            
                rowplotcount+=1
        if len(col_scatter)>0:
            col_cluster=True
            if "col_cluster" in kwargs:
                col_cluster=kwargs["col_cluster"]
            for i, lname in enumerate(col_scatter):
                r=np.array(dfcol[lname])
                col_labels.append(np.amin(r))
                col_labels.append(np.amax(r))
                col_ticks.append(colplotcount)
                col_ticks.append(colplotcount+tickpos)
                
                r=r-np.amin(r)
                r=r/np.amax(r)
                r=0.9*r
                if col_cluster==True:
                    g.ax_col_colors.bar(np.arange(r.shape[0])+0.5,r[leaves_list(g.dendrogram_col.linkage)]+colplotcount,s=scatterpointsize)
                else:
                    g.ax_col_colors.bar(np.arange(r.shape[0])+0.5,r+colplotcount,s=scatterpointsize)
                
                colplotcount+=1
        
        if len(row_bar)>0:
            row_cluster=True
            if "row_cluster" in kwargs:
                row_cluster=kwargs["row_cluster"]
            
            for i, lname in enumerate(row_bar):
                r=np.array(df[lname])
                row_labels.append(np.amin(r))
                row_labels.append(np.amax(r))
                row_ticks.append(rowplotcount)
                row_ticks.append(rowplotcount+tickpos)
                    
                r=r-np.amin(r)
                r=r/np.amax(r)
                r=0.9*r
                if row_cluster==True:
                    g.ax_row_colors.barh(y=np.arange(r.shape[0])+0.5, width=r[leaves_list(g.dendrogram_row.linkage)],left=[rowplotcount]*r.shape[0])
                else:
                    g.ax_row_colors.barh(r, np.arange(r.shape[0])+0.5,left=[rowplotcount]*r.shape[0])
            
                rowplotcount+=1
        if len(col_bar)>0:
            col_cluster=True
            if "col_cluster" in kwargs:
                col_cluster=kwargs["col_cluster"]
            for i, lname in enumerate(col_bar):
                r=np.array(dfcol[lname])
                col_labels.append(np.amin(r))
                col_labels.append(np.amax(r))
                col_ticks.append(colplotcount)
                col_ticks.append(colplotcount+tickpos)

                if col_cluster==True:
                    g.ax_col_colors.scatter(np.arange(r.shape[0])+0.5,r[leaves_list(g.dendrogram_col.linkage)]/(np.amax(r)*1.1)+colplotcount)
                else:
                    g.ax_col_colors.scatter(np.arange(r.shape[0])+0.5,r/(np.amax(r)*1.1)+colplotcount)
                
                colplotcount+=1
        if g.ax_row_colors!=None:
            
            g.ax_row_colors.set_xticks(np.arange(len(_row_colors_title))+0.5)
            g.ax_row_colors.set_xticklabels(_row_colors_title, rotation=90)
        if g.ax_col_colors!=None:
            colax_otherside = g.ax_col_colors.twinx()
            colax_otherside.set_yticks(0.5*(np.arange(len(_col_colors_title))+0.5),labels=_col_colors_title)
            colax_otherside.grid(False)
        if g.ax_col_dendrogram!=None:
            col = g.ax_col_dendrogram.get_position()
            g.ax_col_dendrogram.set_position([col.x0, col.y0, col.width*0.5, col.height*0.5])
        
        if show_plot_labels==True:
            
            rowax_otherside = g.ax_row_colors.twiny()
            rowax_otherside.invert_xaxis()
            rowax_otherside.grid(False)
            rowax_otherside.set_xticks(row_ticks, labels=np.round(row_labels,2), rotation=90, fontsize=8)
            col_ticks=np.array(col_ticks)
            g.ax_col_colors.set_yticks(col_ticks, labels=np.round(col_labels,2), fontsize=8)
        
        legend_num=0
        for _title, colorlut in _row_color_legend.items():
            legendhandles=[]
            for label, color in colorlut.items():
                legendhandles.append(Line2D([0], [0], color=color,linewidth=5, label=label))
            legend1=g.ax_heatmap.legend(handles=legendhandles, 
                                        loc="upper left", 
                                        title=_title,
                                        bbox_to_anchor=(1.15, 1-0.2*legend_num))
            g.ax_heatmap.add_artist(legend1)
            legend_num+=1
        for _title, colorlut in _col_color_legend.items():
            legendhandles=[]
            for label, color in colorlut.items():
                legendhandles.append(Line2D([0], [0], color=color,linewidth=5, label=label))

            legend1=g.ax_heatmap.legend(handles=legendhandles, 
                                        loc="upper left", 
                                        title=_title,
                                        bbox_to_anchor=(1.15, 1-0.2*legend_num))
            g.ax_heatmap.add_artist(legend1)
            legend_num+=1
        
    else:
        g=sns.clustermap(df,method=method,
                         cbar_kws=dict(orientation='horizontal'),
                             xticklabels=xticklabels,
                             cmap=heatmap_palette, 
                             yticklabels=yticklabels,
                             dendrogram_ratio=0.1,
                             figsize=figsize,
                             **kwargs)
    if color_var>0:
        cmap = cm.tab20c(np.linspace(0, 1, color_var))
    else:
        cmap = cm.tab20c(np.linspace(0, 1, approx_clusternum+5))
    hierarchy.set_link_color_palette([mpl.colors.rgb2hex(rgb[:3]) for rgb in cmap])
    
    """coloring the row dendrogram based on branch numbers crossed with the threshold"""
    
    if g.dendrogram_row != None:
        t=_dendrogram_threshold(g.dendrogram_row.dendrogram,approx_clusternum)
               
        den=hierarchy.dendrogram(g.dendrogram_row.linkage,
                                                 labels = g.data.index,
                                                 color_threshold=t,ax=g.ax_row_dendrogram,
                            orientation="left")  
        g.ax_row_dendrogram.invert_yaxis()
        clusters = _get_cluster_classes(den)
        
        keys=list(clusters.keys())
        ckeys={}
        i=1
        for k in keys:
            if k=="C0":
                ckeys[k]="C0"
            else:
                ckeys[k]="C"+str(i)
                i+=1
        for c, v in clusters.items():
            _c=ckeys[c]
            for _v in v:
                cdata["Cluster"].append(_c)
                cdata["Index"].append(_v)
                cdata["RGB"].append(matplotlib.colors.to_rgb(c))
        """Setting the row dendrogram ends here"""
    
    
    """coloring the col dendrogram based on branch numbers crossed with the threshold"""
    col_cdata={"Cluster":[],"Index":[],"RGB":[]}
    if g.dendrogram_col != None:
        t=_dendrogram_threshold(g.dendrogram_col.dendrogram,approx_clusternum_col)
        den=hierarchy.dendrogram(g.dendrogram_col.linkage,
                                                 labels = g.data.columns,
                                                 color_threshold=t,ax=g.ax_col_dendrogram,
                            orientation="top")  
        #g.ax_col_dendrogram.invert_yaxis()
        col_clusters = _get_cluster_classes(den)
        col_cdata={"Cluster":[],"Index":[],"RGB":[]}
        col_keys=list(col_clusters.keys())
        col_ckeys={}
        i=1
        for k in col_keys:
            if k=="C0":
                col_ckeys[k]="C0"
            else:
                col_ckeys[k]="C"+str(i)
                i+=1
        for c, v in col_clusters.items():
            _c=col_ckeys[c]
            for _v in v:
                col_cdata["Cluster"].append(_c)
                col_cdata["Index"].append(_v)
                col_cdata["RGB"].append(matplotlib.colors.to_rgb(c))
    """Setting the col dendrogram ends here"""
    
    
    g.fig.subplots_adjust(bottom=0.175, right=0.70)
    
    x0, _y0, _w, _h = g.cbar_pos
    
    g.ax_cbar.set_title(ctitle)
    #g.ax_cbar.tick_params(axis='x', length=5)
    g.ax_cbar.set_position(pos=[0.80, 0.1, 0.1, 0.02], 
                           which="both")
    if title !="":
        g.fig.suptitle(title, va="bottom")
    plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
   
    _save(save, "complex_clustermap")
    if show:
        plt.show()
    return {"data":g.data2d,"row_clusters":pd.DataFrame(cdata),"col_clusters":pd.DataFrame(col_cdata), "grid":g}
 

def dotplot(df: pd.DataFrame,
            y: str="",
            x: str="",
            dfc=pd.DataFrame(),
            scaling: float=10,
            color_val: str="",
            size_val: str="",
            highlight: str="",
            color_title: str="",
            size_title: str="",
            figsize: list=[],
            save: str="",
            threshold: float=-np.log10(0.05),
            row_clustering: bool=True,
            xtickrotation: float=90,
            column_order: list=[],
            colorpalette="coolwarm",
            show: bool=False,
            title: str="",
            row: str="",
            col: str="",
            ) -> Dict[str, plt.Axes]:
    """
    Drawing a dotplot that can represent two different variables as dot sizes and colors on a regular grid.
    This function is assumed to plot GO enrichment analysis with multiple gene sets.
    
    Parameters
    ----------
    df : pandas DataFrame
        dataframe containing two categories and corresponding values (such as p values and odds ratio).
        e.g.:
            Cluster                   Condensate      pval      odds       FDR
        54       C1                   Cajal body -0.000000  0.000000 -0.000000
        55       C1            *DNA repair focus -0.000000  0.000000 -0.000000
        56       C1  *DNA replication condensate -0.000000  0.000000 -0.000000
        57       C1                       P-body -0.000000  0.000000 -0.000000
        58       C1                     PML body -0.000000  0.000000 -0.000000
    row: string
        the column name of a category that is going to be placed in the row of the dotplot
    col: string
        the column name of a category that is going to be placed in the column of the dotplot
    color_val : string
        The column name for the values represented as dot colors.
    size_val : string
        The column name for the values represented as dot sizes. 
    scaling: float
        The scale of dots. If resulting dots are too large (or small), you can reduce (or increase) dot sizes by adjusting this value.
    highlight : string
        A dictionary to set color labels to leaves. The key is the name of the color label. 
        The value is the list of RGB color codes, each corresponds to the color of a leaf. 
        e.g., {"color1":[[1,0,0,1], ....]}   
    size_title : string
        The title for size values. If not set, "size_val" will be used.
    
    color_title : string
        The title for color values. If not set, "color_val" will be used.
    show : bool
        Whether or not to show the figure.
    Returns
    -------
    axes: dict {"axes1":ax1,"axes2":ax2,"axes3":ax3}
    
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
    if x !="" and y!="":
        row=y
        col=x
    if size_val!="":
        _df=df.pivot_table(index=col,columns=row,values=size_val)
        if len(column_order)>0:
            _df=_df.loc[column_order]
        else:
            _df=_df.sort_index(axis=0,key=natsort_keygen())
        _df=_df.fillna(0)
            
        if color_val!="":
            dfc=df.pivot_table(index=col,columns=row,values=color_val)
            if len(column_order)>0:
                dfc=dfc.loc[column_order]
            else:
                dfc=dfc.sort_index(axis=0,key=natsort_keygen())
            dfc=dfc.fillna(0)
        if highlight !="":
            dfh=df.pivot_table(index=col,columns=row,values=highlight)
            if len(column_order)>0:
                dfh=dfh.loc[column_order]
            else:
                dfh=dfh.sort_index(axis=0,key=natsort_keygen())
            dfh=dfh.fillna(0)
        
        if row_clustering==True:
            Y = fcl.linkage(_df.to_numpy().T, method='ward', metric='euclidean')
            Z1 = sch.dendrogram(Y,no_plot=True)
            idx1 = Z1['leaves']
            reorder=_df.columns[idx1]
            _df=_df[reorder]
            if color_val!="":
                dfc=dfc[reorder]
            if highlight !="":
                dfh=dfh[reorder]
        _x=_df.index
        _y=_df.columns
        mat=_df.to_numpy()
        minsize=np.amin(mat)
        maxsize=np.amax(mat)
    else:
        _x=df.index
        _y=df.columns
        mat=df.to_numpy()
        minsize=np.amin(mat)
        maxsize=np.amax(mat)
    #if minsize==0:
        #mat=mat+maxsize*0.01
    #minsize=np.amin(mat)
    
    maxsize=np.round(maxsize)
    middle0=np.round((minsize+maxsize)/3)
    middle1=np.round(2*(minsize+maxsize)/3)
    #plt.rcParams["figure.figsize"] = [7.50, 3.50]
    #plt.rcParams["figure.autolayout"] = True
    #x = np.arange(len(_x))
    #y = np.arange(len(_y))
    #X, Y = np.meshgrid(x, y)
    xy=[ [i,j] for i in range(len(_x)) for j in range(len(_y))]
    #num = 1000
    sizes = [mat[i,j]*scaling for i in range(len(_x)) for j in range(len(_y))]
    edge_colors=[]
    if highlight !="":
        hmat=dfh.to_numpy()
        hvals = [hmat[i,j]*scaling for i in range(len(_x)) for j in range(len(_y))]
       
        for s in hvals:
            if s>=threshold*scaling:
                edge_colors.append("magenta")
            else:
                edge_colors.append("gray")
    else:
        for s in sizes:
            if s>=threshold*scaling:
                edge_colors.append("magenta")
            else:
                edge_colors.append("gray")
    
    if len(dfc) !=0:
        viridis = cm.get_cmap(colorpalette, 12)
        cmat=dfc.to_numpy()
        cmat[cmat==np.inf]=0
        _cmat=cmat/np.amax(cmat)
        _colors = [viridis(_cmat[i,j]) for i in range(len(_x)) for j in range(len(_y))]
    else:
        _colors = [[0,1,0,1] for i in range(len(_x)) for j in range(len(_y))]
    #print(sizes)
    #xy = 10 * np.random.random((num, 2))
    #xy=XY
    #patches = [plt.Circle(center, size) for center, size in zip(xy, sizes)]
    
    #fig, ax = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [8, 2]})
    if len(figsize)==0:
        figsize=[mat.shape[0]*0.5+2,mat.shape[1]*0.5+1]
    
    fig = plt.figure(figsize=figsize)
    #fig.set_figheight(6)
    #fig.set_figwidth(6)
    
    ax1 = plt.subplot2grid(shape=(10, 6), loc=(0, 0), colspan=4, rowspan=10)
    ax2 = plt.subplot2grid(shape=(10, 6), loc=(1, 4), colspan=2, rowspan=4)
    ax3 = plt.subplot2grid(shape=(10, 6), loc=(6, 4), colspan=2, rowspan=1)
 
    collection = mc.CircleCollection(sizes,
                                     edgecolors=edge_colors, 
                                     offsets=xy, 
                                     transOffset=ax1.transData, 
                                     facecolors=_colors,
                                     linewidths=2)
    ax1.add_collection(collection)
    ax1.margins(0.1)
    ax1.set_xlim(-0.5,len(_x)-0.5)
    ax1.set_xticks(np.arange(len(_x)))
    ax1.set_xticklabels(_x,rotation=xtickrotation)
    ax1.set_yticks(np.arange(len(_y)))
    ax1.set_yticklabels(_y, rotation=0)
    if color_title=="":
        color_title=color_val
    
    if len(dfc) !=0:
        norm = mpl.colors.Normalize(vmin=np.min(cmat), vmax=np.amax(cmat))
        
        cb1 = mpl.colorbar.ColorbarBase(ax3, cmap=viridis,
                                        norm=norm,
                                        orientation='horizontal')
        cb1.set_label(color_title)
    #ax[1]=fig.add_axes([1,0.3,0.1,1])
    
    lxy=[[0.5, i*0.5] for i in range(3)]
    collection2 = mc.CircleCollection([middle0*scaling,middle1*scaling, maxsize*scaling], 
                                      offsets=lxy, 
                                      transOffset=ax2.transData, 
                                      facecolors='lightgray',
                                      edgecolors="gray")
    ax2.add_collection(collection2)
    ax2.axis('off')
    ax2.margins(0.3)
    for text, (x, y) in zip([middle0,middle1, maxsize], lxy):
        ax2.text(x+0.01, y,str(text), ha="left",va="center",color="black" )
    if size_title=="":
        size_title=size_val
    ax2.text(0.5,-0.5, size_title,va="center",ha="center")
    #ax[1].set_yticks(np.arange(3))
    #ax[1].set_yticklabels([minsize,middle, maxsize], rotation=0)
    #plt.tight_layout()
    plt.subplots_adjust(left=0.3,bottom=0.2)
    #plt.tight_layout()
    if title !="":
        fig.suptitle(title)
    _save(save, "dotplot")
    if show==True:
        plt.show()
    return {"axes1":ax1,"axes2":ax2,"axes3":ax3}


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
        
    _save(save, "violin")
    
    return {"p values":newpvals,"axes":ax}



def volcanoplot():
    pass


if __name__=="__main__":
    pass