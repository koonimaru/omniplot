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
from natsort import natsort_keygen
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
from omniplot.utils import _separate_data, _line_annotate, _dendrogram_threshold, _radialtree2,_get_cluster_classes,_calc_curveture, _draw_ci_pi,_calc_r2,_ci_pi, _save, _baumkuchen_xy, _get_embedding
import scipy.stats as stats
from joblib import Parallel, delayed
from omniplot.chipseq_utils import _calc_pearson
import itertools as it

colormap_list: list=["nipy_spectral", "terrain","tab20b","tab20c","gist_rainbow","hsv","CMRmap","coolwarm","gnuplot","gist_stern","brg","rainbow","jet"]
hatch_list: list = ['//', '\\\\', '||', '--', '++', 'xx', 'oo', 'OO', '..', '**','/o', '\\|', '|*', '-\\', '+o', 'x*', 'o-', 'O|', 'O.', '*-']
maker_list: list=['.', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'D', 'd', 'P', 'X','o', '1', '2', '3', '4', '+', 'x', '|', '_']

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
               **kwargs) -> Dict:
    """
    Drawing a radial dendrogram with color labels.
    
    Parameters
    ----------
    df : pandas DataFrame
        A wide format data. 
        
        
    n_clusters: int
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
    Z=sch.dendrogram(Y,
                        labels = _labels,
                        color_threshold=t,no_plot=True)
    sample_classes={k: list(category_df[k]) for k in category_df.columns}
    ax=_radialtree2(Z, sample_classes=sample_classes,addlabels=False, **kwargs)
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
                title: str="",):
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
        dmat=Parallel(n_jobs=-1)(delayed(_calc_pearson)(ind, X) for ind in list(it.combinations(range(X.shape[0]), 2)))
        dmat=np.array(dmat)
        dmat=squareform(dmat)
        print(dmat)
        dmat+=np.identity(dmat.shape[0])
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
        return res
    else:
        
        g=sns.clustermap(data=dmat,
                         xticklabels=xticklabels,
                         yticklabels=yticklabels,
                   method="ward", 
                   cmap=palette,
                   col_cluster=True,
                   row_cluster=True,
                   figsize=figsize,
                   rasterized=True,
                    #cbar_kws={"label":"Pearson correlation"}, 
                   annot=show_values,
                   **clustermap_param)
        
        g.cax.set_ylabel(ctitle, rotation=-90,va="bottom")
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)  # For y axis
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90) # For x axis
        return {"grid":g}

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
                       save: str="",heatmap_col: list=[],
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
        dict: {"row_clusters":pd.DataFrame,"col_clusters":pd.DataFrame, "grid":g}
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
        xsize=np.amin([2*cnum, 20])
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
                
                u=np.unique(df[k])
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
                u=np.unique(dfcol[k])
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
                             figsize=figsize,dendrogram_ratio=0.1,cbar_kws={"label":ctitle},
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
                             figsize=figsize,cbar_kws={"label":ctitle},**kwargs)
            g.ax_col_colors.invert_yaxis()
        elif len(_row_colors) >0:
            g=sns.clustermap(df[variables],row_colors=_row_colors,method=method,cbar_kws={"label":ctitle},xticklabels=xticklabels, yticklabels=yticklabels,dendrogram_ratio=0.1,figsize=figsize,**kwargs)
            g.ax_row_colors.invert_xaxis()
        
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

            legend1=g.ax_heatmap.legend(handles=legendhandles, loc=[1.15,0.8-0.2*legend_num], title=_title)
            g.ax_heatmap.add_artist(legend1)
            legend_num+=1
        for _title, colorlut in _col_color_legend.items():
            legendhandles=[]
            for label, color in colorlut.items():
                legendhandles.append(Line2D([0], [0], color=color,linewidth=5, label=label))

            legend1=g.ax_heatmap.legend(handles=legendhandles, loc=[1.15,0.8-0.2*legend_num], title=_title)
            g.ax_heatmap.add_artist(legend1)
            legend_num+=1
        
    else:
        g=sns.clustermap(df,method=method,cbar_kws={"label":ctitle},**kwargs)
    if color_var>0:
        cmap = cm.nipy_spectral(np.linspace(0, 1, color_var))
    else:
        cmap = cm.nipy_spectral(np.linspace(0, 1, approx_clusternum+5))
    hierarchy.set_link_color_palette([mpl.colors.rgb2hex(rgb[:3]) for rgb in cmap])
    
    """coloring the row dendrogram based on branch numbers crossed with the threshold"""
    if g.dendrogram_row != None:
        t=_dendrogram_threshold(g.dendrogram_row.dendrogram,approx_clusternum)
        # lbranches=np.array(g.dendrogram_row.dendrogram["dcoord"])[:,:2]
        # rbranches=np.array(g.dendrogram_row.dendrogram["dcoord"])[:,2:]
        # thre=np.linspace(0, np.amax(g.dendrogram_row.dendrogram["dcoord"]), 100)[::-1]
        # for t in thre:
        #     #print(np.sum(lbranches[:,1]>t),np.sum(rbranches[:,0]>t),np.sum(lbranches[:,0]>t),np.sum(rbranches[:,1]>t))
        #     crossbranches=np.sum(lbranches[:,1]>t)+np.sum(rbranches[:,0]>t)-np.sum(lbranches[:,0]>t)-np.sum(rbranches[:,1]>t)
        #     #print(crossbranches)
        #
        #     if crossbranches>approx_clusternum:
        #         break
        
        den=hierarchy.dendrogram(g.dendrogram_row.linkage,
                                                 labels = g.data.index,
                                                 color_threshold=t,ax=g.ax_row_dendrogram,
                            orientation="left")  
        g.ax_row_dendrogram.invert_yaxis()
        clusters = _get_cluster_classes(den)
        cdata={"Cluster":[],"Index":[],"RGB":[]}
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
    if title !="":
        g.fig.suptitle(title, va="bottom")
    plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    plt.subplots_adjust(bottom=0.165, right=0.75)
    _save(save, "complex_clustermap")
    if show:
        plt.show()
    if return_col_cluster==True:
        return {"data":g.data2d,"row_clusters":pd.DataFrame(cdata),"col_clusters":pd.DataFrame(col_cdata), "grid":g}
    else:
        return {"data":g.data2d,"row_clusters":pd.DataFrame(cdata),"col_clusters":None, "grid":g}

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


def violinplot(df, 
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


# proportion plots



def _stacked_barplot(df: pd.DataFrame,
                    x: Union[str, list],
                    hue: Union[str, list],
                    scale: str="fraction",
                    order: list=[],
                    hue_order: list=[],
                    test_pairs: List[List[str]]=[],
                    show_values: bool=True,
                    show: bool=False,
                    figsize: List[int]=[4,6],
                    xunit: str="",
                    yunit: str="",
                    title: str="",
                    hatch: bool=False)-> Dict:
    
    """
    Drawing a stacked barplot with or without the fisher's exact test 
    
    Parameters
    ----------
    df : pandas DataFrame
    
    x: str or list
        The category to place in x axis. Only str values are accepted.
    hue: str or list
        Counting samples by the hue category. Only str values are accepted.
    order: list, optional
        The order of x axis labels
    hue_order: list, optional
        The order of hue labels
    scale: str, optional
        Scaling method. Available options are: fraction, percentage, absolute
    test_pairs : pairs of categorical values related to x. It will calculate -log10 (p value) (mlp) of the fisher exact test.
        Examples: [["Adelie","Chinstrap" ],
                    ["Gentoo","Chinstrap" ],
                    ["Adelie","Gentoo" ]]
    show_values: bool, optional
        Wheter to exhibit the values of fractions/counts/percentages.
    
    show : bool, optional
        Whether or not to show the figure.
    
    figsize : List[int], optional
        The figure size, e.g., [4, 6].
     
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
    
    data: Dict={}
    
    if df[x].isnull().values.any():
        df[x]=df[x].replace(np.nan, "NA")
    
    if df[hue].isnull().values.any():
        df[hue]=df[hue].replace(np.nan, "NA")
    
    
    if len(order)==0:
        u=np.unique(df[x])
        keys=sorted(list(u))
    else:
        keys=order
    if len(hue_order)==0:
        u=np.unique(df[hue])
        hues=sorted(list(u))
    else:
    
        hues=hue_order
    for key in keys:
        data[key]=[]
        for h in hues:
            data[key].append(np.sum((df[x]==key) & (df[hue]==h)))
    pvals={}
    if len(test_pairs) >0:
        
        for i, h in enumerate(hues):
            pvals[h]=[]
            for p1,p2 in test_pairs:
                idx1=keys.index(p1)
                idx2=keys.index(p2)
                yes_total=np.sum(data[keys[idx1]])
                no_total=np.sum(data[keys[idx2]])
                yes_and_hue=data[keys[idx1]][i]
                no_and_hue=data[keys[idx2]][i]
                table=[[yes_and_hue, no_and_hue],
                       [yes_total-yes_and_hue, no_total-no_and_hue]]
                
                odd, pval=fisher_exact(table)
                pvals[h].append([idx1, idx2, pval])
    if scale=="fraction":
        for key in keys:
            data[key]=np.array(data[key])/np.sum(data[key])
    elif scale=="percentage":
        for key in keys:
            data[key]=np.array(data[key])/np.sum(data[key])*100
    bottom=np.zeros([len(keys)])
    cmap=plt.get_cmap("tab20b")
    fig, ax=plt.subplots(figsize=figsize)
    plt.subplots_adjust(left=0.2,right=0.6, bottom=0.17)
    if scale=="absolute":
        unit=""
    elif scale=="fraction":
        unit=""
    elif scale=="percentage":
        unit="%"
    pos={}
    #_hatch_list=[hatch_list[i] for i in range(len(keys))]
    for i, h in enumerate(hues):
        
        heights=np.array([data[key][i] for key in keys])
        
        if hatch==True:
            plt.bar(keys, heights,width=0.5, bottom=bottom, color=cmap(i/len(hues)), label=h, hatch=hatch_list[i])
        else:
            plt.bar(keys, heights,width=0.5, bottom=bottom, color=cmap(i/len(hues)), label=h)
        if show_values==True:
            for j in range(len(keys)):
                if scale=="absolute":
                    plt.text(j,bottom[j]+heights[j]/2,"{}{}".format(heights[j],unit), 
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="y", lw=1, alpha=0.8))
                else:
                    plt.text(j,bottom[j]+heights[j]/2,"{:.2f}{}".format(heights[j],unit), 
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="y", lw=1, alpha=0.8))
        plt.xticks(rotation=90)
        pos[h]={key: [he, bo] for key, he, bo in zip(keys, heights, bottom)}
        bottom+=heights
    ax.legend(loc=[1.01,0])
    ax.set_xlabel(x)
    if scale=="absolute":
        ylabel="Counts"
    elif scale=="fraction":
        ylabel="Fraction"
    elif scale=="percentage":
        ylabel="Percentage"
    ax.set_ylabel(ylabel)
    
    if len(pvals)>0:
        print("mlp stands for -log10(p value)")
        for i, h in enumerate(hues):
            _pos=pos[h]
            for idx1, idx2, pval in pvals[h]:
                if pval < 0.05:
                    he1, bot1=_pos[keys[idx1]]
                    he2, bot2=_pos[keys[idx2]]
                    line, =plt.plot([idx1,idx2],[he1/2+bot1,he2/2+bot2],color="gray")
                    # r1=ax.transData.transform([idx1, he1/2+bot1])
                    # r2=ax.transData.transform([idx2, he2/2+bot2])
                    r1=np.array([idx1, he1/2+bot1])
                    r2=np.array([idx2, he2/2+bot2])
                    r=r2-r1
                    print(ax.get_xlim(),ax.get_ylim())
                    r=np.array([1,3])*r/np.array([ax.get_xlim()[1]-ax.get_xlim()[0],ax.get_ylim()[1]-ax.get_ylim()[0]])
                    #r=ax.transData.transform(r)
                    if idx2<idx1:
                        r=-r
                    print(r)
                    r=r*(r @ r)**(-0.5)
                    print(h,r)
                    angle=np.arccos(r[0])
                    if r[1]<0:
                        angle= -angle
                    print(angle)
                    _line_annotate( "mlp="+str(np.round(-np.log10(pval), decimals=1)), line, (idx1+idx2)/2, color="magenta")
                    # plt.text((idx1+idx2)/2, 0.5*(he1/2+bot1+he2/2+bot2), "mlp="+str(np.round(-np.log10(pval), decimals=1)), 
                    #          color="magenta", va="center",ha="center", rotation=360*angle/(2*np.pi),)
                    # plt.annotate("mlp="+str(np.round(-np.log10(pval), decimals=1)),[(r1[0]+r2[0])/2, 0.5*(r1[1]+r2[1])],   
                    #          color="magenta",ha="center", rotation=360*angle/(2*np.pi),xycoords='figure pixels')
                    #
    if show:
        plt.show()
    if title !="":
        fig.suptitle(title)
    return {"pval":pvals,"axes":ax}


def stacked_barplot(df: pd.DataFrame,
                    x: Union[str, list],
                    hue: Union[str, list],
                    scale: str="fraction",
                    order: Optional[list]=None,
                    hue_order: Optional[list]=None,
                    test_pairs: List[List[str]]=[],
                    palette: Union[str,dict]="tab20c",
                    show_values: bool=True,
                    show: bool=False,
                    figsize: List[int]=[],
                    xunit: str="",
                    yunit: str="",
                    title: str="",
                    hatch: bool=False,
                    rotation: int=90,
                    ax: Optional[plt.Axes]=None,
                    show_legend:bool=True)-> Dict:
    
    """
    Drawing a stacked barplot with or without the fisher's exact test 
    
    Parameters
    ----------
    df : pandas DataFrame
    
    x: str or list
        The category to place in x axis. Multiple categories can be passed by a list.
    hue: str or list
        Counting samples by the hue category. Multiple categories can be passed by a list.
    order: list, optional
        The order of x axis labels
    hue_order: list, optional
        The order of hue labels
    scale: str, optional
        Scaling method. Available options are: fraction, percentage, absolute
    test_pairs : pairs of categorical values related to x. It will calculate -log10 (p value) (mlp) of the fisher exact test.
        Examples: [["Adelie","Chinstrap" ],
                    ["Gentoo","Chinstrap" ],
                    ["Adelie","Gentoo" ]]
    show_values: bool, optional
        Wheter to exhibit the values of fractions/counts/percentages.
    
    show : bool, optional
        Whether or not to show the figure.
    
    figsize : List[int], optional
        The figure size, e.g., [4, 6].
     
    Returns
    -------
    dict {"pval":pvals,"axes":ax}
    
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
    
    if type(x)==str:
        x=[x]
        if order!=None:
            order=[order]
    if type(hue)==str:
        hue=[hue]
        if hue_order!=None:
            hue_order=[hue_order]
    for _x in x:
        if df[_x].isnull().values.any():
            df[_x]=df[_x].replace(np.nan, "NA")
    for _hue in hue:
        if df[_hue].isnull().values.any():
            df[_hue]=df[_hue].replace(np.nan, "NA")

    xkeys={}
    keysx={}
    meankey_len=0
    for i, _x in enumerate(x):
        if order==None:
            u=np.unique(df[_x])
            keys=sorted(list(u))
        else:
            keys=order[i]
        meankey_len+=len(keys)
        xkeys[_x]=keys
        for k in keys:
            keysx[k]=_x
    meankey_len=meankey_len//len(x)
    huekeys={}
    for i, _hue in enumerate(hue):
        if hue_order==None:
            u=np.unique(df[_hue])
            hues=sorted(list(u))
        else:
        
            hues=hue_order[i]
        huekeys[_hue]=hues
    
    data={}
    for _x, keys in xkeys.items():
        data[_x]={}
        for key in keys:
            
            for _hue, hues in  huekeys.items():
                if _x==_hue:
                    continue
                if not _hue in data[_x]:
                    data[_x][_hue]={}
                data[_x][_hue][key]=[]
                for h in hues:
                    data[_x][_hue][key].append(np.sum((df[_x]==key) & (df[_hue]==h)))

    
    pvals={}
    if len(test_pairs) >0:
        print("mlp stands for -log10(p value)")
        for _hue in hue:
            
            for i, h in enumerate(huekeys[_hue]):
                
                for p1,p2 in test_pairs:
                    _x=keysx[p1]
                    __x=keysx[p2]
                    if _x!=__x:
                        raise Exception("{} and {} can not be compared.".format(p1, p2))
                    
                    if _x==_hue:
                        continue
                    
                    if not _x in pvals:
                        pvals[_x]={}
                    if not _hue in pvals[_x]:
                        pvals[_x][_hue]={}
                    if not h in pvals[_x][_hue]:
                        pvals[_x][_hue][h]=[]
                    keys=xkeys[_x]
                    idx1=xkeys[_x].index(p1)
                    idx2=xkeys[_x].index(p2)
                    yes_total=np.sum(data[_x][_hue][keys[idx1]])
                    no_total=np.sum(data[_x][_hue][keys[idx2]])
                    yes_and_hue=data[_x][_hue][keys[idx1]][i]
                    no_and_hue=data[_x][_hue][keys[idx2]][i]
                    table=[[yes_and_hue, no_and_hue],
                           [yes_total-yes_and_hue, no_total-no_and_hue]]
        
                    odd, pval=fisher_exact(table)
                    pvals[_x][_hue][h].append([idx1, idx2, pval])
    if scale=="fraction":
        for _x in x:
            for _hue in hue:
                if _x==_hue:
                    continue
                for key in keys:
                    data[_x][_hue][key]=np.array(data[_x][_hue][key])/np.sum(data[_x][_hue][key])
    elif scale=="percentage":
        for _x in x:
            for _hue in hue:
                if _x==_hue:
                    continue
                for key in keys:
                    data[_x][_hue][key]=np.array(data[_x][_hue][key])/np.sum(data[_x][_hue][key])*100
    if type(palette)==str:
        cmap=plt.get_cmap(palette)
        
    ncols=len(x)*len(hue)-len(set(x)&set(hue))
    if ncols<1:
        ncols=1
    if len(figsize)==0:
        figsize=[(4+0.1*meankey_len)*ncols, 6]
    if ax!=None:
        axes=ax
        fig=None
    else:
        fig, axes=plt.subplots(figsize=figsize,ncols=ncols)
    
    if ncols==1:
        axes=[axes]
    else:
        axes=axes.flatten()
    
    if scale=="absolute":
        unit=""
    elif scale=="fraction":
        unit=""
    elif scale=="percentage":
        unit="%"
    axindex=0
    pos={}
    #_hatch_list=[hatch_list[i] for i in range(len(keys))]
    for _x in x:
        pos[_x]={}
        for _hue in hue:
            if _x==_hue:
                continue
            pos[_x][_hue]={}
            keys=xkeys[_x]
            hues=huekeys[_hue]
            bottom=np.zeros([len(keys)])
            for i, h in enumerate(hues):
                ax=axes[axindex]
                
                heights=np.array([data[_x][_hue][key][i] for key in keys])
                
                if type(palette)==dict:
                    if hatch==True:
                        ax.bar(keys, heights,width=0.5, bottom=bottom, color=palette[h], label=h, hatch=hatch_list[i])
                    else:
                        ax.bar(keys, heights,width=0.5, bottom=bottom, color=palette[h], label=h)
                else:
                    if hatch==True:
                        ax.bar(keys, heights,width=0.5, bottom=bottom, color=cmap(i/len(hues)), label=h, hatch=hatch_list[i])
                    else:
                        ax.bar(keys, heights,width=0.5, bottom=bottom, color=cmap(i/len(hues)), label=h)
                if show_values==True:
                    for j in range(len(keys)):
                        if scale=="absolute":
                            ax.text(j,bottom[j]+heights[j]/2,"{}{}".format(heights[j],unit), 
                                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="y", lw=1, alpha=0.8))
                        else:
                            ax.text(j,bottom[j]+heights[j]/2,"{:.2f}{}".format(heights[j],unit), 
                                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="y", lw=1, alpha=0.8))
                
                pos[_x][_hue][h]={key: [he, bo] for key, he, bo in zip(keys, heights, bottom)}
                bottom+=heights
            ax.set_xticks(ax.get_xticks(), labels=keys, rotation=rotation)
            #ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            if show_legend==True:
                ax.legend(loc=[1.01,0])
            ax.set_xlabel(_x)
            if scale=="absolute":
                ylabel="Counts"
            elif scale=="fraction":
                ylabel="Fraction"
            elif scale=="percentage":
                ylabel="Percentage"
            ax.set_ylabel(ylabel)
            axindex+=1
            if len(pvals)>0 and _x in pvals:
                
                for _hue in hue:
                    if _x==_hue:
                        continue
                    if not _hue in pos[_x]:
                        continue
                    hues=huekeys[_hue]
                    for i, h in enumerate(hues):
                        #print(pos)
                        #print(pos[_x])
                        _pos=pos[_x][_hue][h]
                        for idx1, idx2, pval in pvals[_x][_hue][h]:
                            
                            he1, bot1=_pos[keys[idx1]]
                            he2, bot2=_pos[keys[idx2]]
                            line, =ax.plot([idx1,idx2],[he1/2+bot1,he2/2+bot2],color="gray")
                            # r1=ax.transData.transform([idx1, he1/2+bot1])
                            # r2=ax.transData.transform([idx2, he2/2+bot2])
                            r1=np.array([idx1, he1/2+bot1])
                            r2=np.array([idx2, he2/2+bot2])
                            r=r2-r1
                            #print(ax.get_xlim(),ax.get_ylim())
                            r=np.array([1,3])*r/np.array([ax.get_xlim()[1]-ax.get_xlim()[0],ax.get_ylim()[1]-ax.get_ylim()[0]])
                            #r=ax.transData.transform(r)
                            if idx2<idx1:
                                r=-r
                            #print(r)
                            r=r*(r @ r)**(-0.5)
                            #print(h,r)
                            angle=np.arccos(r[0])
                            if r[1]<0:
                                angle= -angle
                            #print(angle)
                            if pval < 0.05:
                                pval_str=str(np.round(-np.log10(pval), decimals=1))
                            else:
                                pval_str="ns"
                            _line_annotate( "mlp="+pval_str, line, (idx1+idx2)/2, color="magenta")
    if len(x)==1 and len(hue)==1:
        plt.subplots_adjust(top=0.93,left=0.1,right=0.7, bottom=0.17)                            
    else:
        plt.tight_layout(w_pad=2)
    
    if title !="" and fig !=None:
        fig.suptitle(title)
    if show:
        plt.show()
    return {"pval":pvals,"axes":ax}


def nice_piechart(df: pd.DataFrame, 
                  category: Union[str, List[str]],
                  palette: str="tab20c",
                  ncols: int=2,
                  ignore: float=0.05,
                  show_values: bool=True,title: str="",) ->Dict:
    
    if type(category)==str:
        category=[category]
    nrows=len(category)//ncols+int(len(category)%ncols!=0)
    fig, axes=plt.subplots(nrows=nrows, ncols=ncols, figsize=[ncols*2,
                                                        nrows*2])
    axes=axes.flatten()
    for cat, ax in zip(category, axes):
        u, c=np.unique(df[cat], return_counts=True)
        
        srt=np.argsort(c)[::-1]
        u=u[srt]
        c=c[srt]
        _c=c/np.sum(c)
        
        
        
        _cmap=plt.get_cmap(palette, c.shape[0])
        colors=[_cmap(i) for i in range(c.shape[0])]
        for j in range(c.shape[0]):
            if _c[j]<ignore:
                colors[j]=[0,0,0,1]
                u[j]=""
                continue
            if show_values==True:
                u[j]=u[j]+"\n("+str(100*np.round(_c[j],1))+"%)"
        
        ax.pie(c, labels=u, 
               counterclock=False,
               startangle=90, 
               colors=colors,
               labeldistance=0.6,
               radius=1.25)
        ax.set_title(cat,backgroundcolor='lavender',pad=10)
    if len(category)%ncols!=0:
        for i in range(len(category)%ncols-2):
            fig.delaxes(axes[-(i+1)])
    plt.tight_layout(h_pad=1)
    plt.subplots_adjust(top=0.9)
    return {"axes":ax}


def nice_piechart_num(df: pd.DataFrame,hue: List[str],
                      category: str="" ,
                  
                  palette: str="tab20c",
                  ncols: int=2,
                  ignore: float=0.05,
                  show_values: bool=True,
                  title: str="",
                  figsize=[]) ->Dict:
    
    if category=="":
        category=list(df.index)
    else:
        df=df.set_index(category)
        category=list(df.index)
        
    df=df[hue]
    srt=np.argsort(df.sum(axis=0))[::-1]
    df=df[df.columns[srt]]
    hue=list(df.columns)
    nrows=len(category)//ncols+int(len(category)%ncols!=0)
    if len(figsize)==0:
        figsize=[ncols*2,nrows*2]
    fig, axes=plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes=axes.flatten()
    _cmap=plt.get_cmap(palette, len(hue))
    colors=[_cmap(i) for i in range(len(hue))]
    for cat, ax in zip(category, axes):
        c=df.loc[cat]
        _c=c/np.sum(c)
        ax.pie(c, 
               counterclock=False,
               startangle=90, 
               colors=colors,
               labeldistance=0.6,
               radius=1.25)
        ax.set_title(cat,backgroundcolor='lavender',pad=8)
    if len(category)%ncols!=0:
        for i in range(len(category)%ncols-2):
            fig.delaxes(axes[-(i+1)])
    plt.tight_layout(h_pad=1)
    plt.subplots_adjust(top=0.95, right=0.81)
    
    legend_elements = [Line2D([0], [0], marker='o', color='lavender', label=huelabel,markerfacecolor=color, markersize=10)
                      for color, huelabel in zip(colors, hue)]
    
    fig.legend(handles=legend_elements,bbox_to_anchor=(1, 1))
    return {"axes":ax}


def stackedlines(df: pd.DataFrame, 
                x: str,
                y: list,
                sort: bool=True,
                inverse: bool=False,
                show_values: bool=False,
                remove_all_zero: bool=False,
                palette: str="tab20c",
                figsize=[7,4],
                ax: Optional[plt.Axes]=None,
                alpha: float=0.75,
                bbox_to_anchor: list=[1.7, 1],
                right: float=0.7,
                bottom: float=0.120,
                show_legend: bool=True,
                xlabel: str="",
                ylabel: str="",
                yunit: str="",
                xunit: str="",
                title: str="",
                hatch: bool=False):
    """
    Drawing a scatter plot of which points are represented by pie charts. 
    
    Parameters
    ----------
    df : pandas DataFrame
        A wide form dataframe. Index names are used to label points. It accepts negative values, but not recommended as 
        it does not make sense.
        e.g.) 
              year    biofuel_consumption    coal_consumption    gas_consumption    hydro_consumption    nuclear_consumption    oil_consumption
        90    1990                 16.733            5337.998           5170.609              864.271               1723.004           9306.913
        91    1991                 19.389            5287.613           5283.972              849.620               1829.645           9108.509
        92    1992                 22.045            5324.031           5463.509              743.463               1848.197           9297.387
        93    1993                 25.759            5522.452           5599.419              825.742               1822.853           9376.045
        94    1994                 28.846            5543.144           5731.081              766.870               1912.903           9619.746
        95    1995                 30.942            5593.053           5979.829              920.274               2011.356           9597.527            
    x : str
        the name of a column to be the x axis of the plot.
        
    y: list
        the names of values to display as stacked lines
    sort: bool, optional (default: True)
        Whether to sort lines based on their values
    show_values: bool, optional (default: False)
        Whether to show percentages at the end of lines.
    
    remove_all_zero: bool, optional (default: False)    
    
    pie_palette : str
        A colormap name
    xlabel: str, optional
        x axis label
    ylabel: str, optional
        y axis label
    ax: Optional[plt.Axes] optional, (default: None)
        pyplot ax to add this scatter plot


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
    df=df.fillna(0)
    X=np.array(df[x])
    Y=[]
    for col in y:
        Y.append(np.array(df[col]))
    Y=np.array(Y)
    if remove_all_zero==True:
        _filter=Y.sum(axis=0)!=0
        Y=Y[:,_filter]
        X=X[_filter]
    Ydict={col:[] for col in y}
    if sort==True:
        
        if inverse==True:
            for i, _x in enumerate(X):
                _Y=Y[:,i]
                srtidx=np.argsort(_Y)[::-1]
                _bottom=0
                _bottom_neg=0
                for _idx in srtidx:
                    _col=y[_idx]
                    yval=Y[_idx,i]
                    if yval>=0:
                        Ydict[_col].append([_bottom, yval+_bottom])
                        _bottom+=yval
                    else:
                        Ydict[_col].append([_bottom_neg, yval+_bottom_neg])
                        _bottom_neg+=yval
        else:
            for i, _x in enumerate(X):
                _Y=Y[:,i]
                srtidx=np.argsort(_Y)
                _bottom=0
                _bottom_neg=0
                for _idx in srtidx:
                    _col=y[_idx]
                    yval=Y[_idx,i]
                    if yval>=0:
                        Ydict[_col].append([_bottom, yval+_bottom])
                        _bottom+=yval
                    else:
                        Ydict[_col].append([_bottom_neg, yval+_bottom_neg])
                        _bottom_neg+=yval
    else:
        for i, _x in enumerate(X):
            _Y=Y[:,i]
            _bottom=0
            _bottom_neg=0
            for _idx,_col in enumerate(y):
                yval=Y[_idx,i]
                if yval>=0:
                    Ydict[_col].append([_bottom, yval+_bottom])
                    _bottom+=yval
                else:
                    Ydict[_col].append([_bottom_neg, yval+_bottom_neg])
                    _bottom_neg+=yval
    if ax ==None:
        fig, ax=plt.subplots(figsize=figsize)
        
    cmap=plt.get_cmap(palette, len(y))
    colorlut={col: cmap(i) for i, col in enumerate(y)}
    last_vals=[]
    last_pos=[]
    i=0
    for col, vals in Ydict.items():
        vals=np.array(vals)
        if hatch==True:
            ax.fill_between(X, vals[:,0], vals[:,1], label=col, alpha=alpha, color=colorlut[col], hatch=hatch_list[i])
        else:
            ax.fill_between(X, vals[:,0], vals[:,1], label=col, alpha=alpha, color=colorlut[col])
        last_vals.append(vals[-1,1]-vals[-1,0])
        last_pos.append(vals[-1,1]/2+vals[-1,0]/2)
        i+=1
    if show_values==True:
        last_vals=100*np.array(last_vals)/np.sum(last_vals)
        for val, pos in zip(last_vals, last_pos):
            ax.text(X[-1], pos, str(np.round(val, 1))+"%")
            
    if show_legend==True:
        plt.legend(bbox_to_anchor=bbox_to_anchor)
    plt.subplots_adjust(right=right, bottom=bottom)
    if xlabel !="":
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if yunit!="":
        ax.text(0, 1, "({})".format(yunit), transform=ax.transAxes, ha="right")
    if xunit!="":
        ax.text(1, 0, "({})".format(xunit), transform=ax.transAxes, ha="left",va="top")
    if inverse==True:
        ax.invert_yaxis()


# scatter plots
        
def decomplot(df: pd.DataFrame,
              variables: List=[],
              category: Union[List, str]="", 
              method: str="pca", 
              component: int=3,
              arrow_color: str="yellow",
              arrow_text_color: str="black",
              show: bool=False, 
              explained_variance: bool=True,
              arrow_num: int=3,
              figsize=[],
              regularization: bool=True,
              pcapram={"random_state":0},
              nmfparam={"random_state":0},
              save: str="",
              title: str="",
              barrierfree: bool=True,
              saveparam: dict={},
              ax: Optional[plt.Axes]=None,) :
    
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
    x, category=_separate_data(df, variables=variables, category=category)
    # if category !="":
    #     category_val=df[category].values
    #     df=df.drop([category], axis=1)
    #     x = df.values
    #     assert x.dtype==float, f"data must contain only float values except {category} column."
    #
    # else:    
    #     x = df.values
    #     assert x.dtype==float, "data must contain only float values."
    original_index=df.index
    if len(variables)!=0:
        features=variables
    else:
        
        features=sorted(list(set(df.columns) - set(category)))
    dfpc_list=[]
    comb=list(combinations(np.arange(component), 2))
        
    if len(category)!=0:
        figures={}
        for cat in category:
            if len(comb)==1:
                fig, axes=plt.subplots()
                axes=[axes]
            else:
                nrows=len(comb)//2+int(len(comb)%2!=0)
                if len(figsize)==0:
                    figsize=[8,3*nrows]
                
                fig, axes=plt.subplots(ncols=2, nrows=nrows, figsize=figsize)
                plt.subplots_adjust(top=0.9,right=0.8)
                axes=axes.flatten()
            figures[cat]={"fig": fig, "axes":axes}
    else:
        figures={}
        if len(comb)==1:
            fig, axes=plt.subplots()
            axes=[axes]
        else:
            nrows=len(comb)//2+int(len(comb)%2!=0)
            if len(figsize)==0:
                figsize=[8,3*nrows]
            
            fig, axes=plt.subplots(ncols=2, nrows=nrows, figsize=figsize)
            plt.subplots_adjust(top=0.9,right=0.8)
            axes=axes.flatten()
        figures["nocat"]={"fig": fig, "axes":axes}
    if method=="pca":
        if regularization:
            x=zscore(x, axis=0)
        pca = PCA(n_components=component,**pcapram)
        pccomp = pca.fit_transform(x)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        combnum=0
        for axi, (i, j) in enumerate(comb):
            xlabel, ylabel='pc'+str(i+1), 'pc'+str(j+1)
            dfpc = pd.DataFrame(data = np.array([pccomp[:,i],pccomp[:,j]]).T, columns = [xlabel, ylabel],index=original_index)
            _loadings=np.array([loadings[:,i],loadings[:,j]]).T
            a=np.sum(_loadings**2, axis=1)
            srtindx=np.argsort(a)[::-1][:arrow_num]
            _loadings=_loadings[srtindx]
            _features=np.array(features)[srtindx]
            
            if len(category)!=0:
                for cat in category:
                    dfpc[cat]=df[cat]
                    if combnum==1:
                        sns.scatterplot(data=dfpc, x=xlabel, y=ylabel, hue=cat, ax=figures[cat]["axes"][axi])
                        figures[cat]["axes"][axi].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
                    else:
                        sns.scatterplot(data=dfpc, x=xlabel, y=ylabel, hue=cat, ax=figures[cat]["axes"][axi],
                                        legend=False)
                    for k, feature in enumerate(_features):
                        figures[cat]["axes"][axi].arrow(0, 0, _loadings[k, 0],_loadings[k, 1],color=arrow_color,width=0.005,head_width=0.1)
                        figures[cat]["axes"][axi].text(_loadings[k, 0],_loadings[k, 1],feature,color=arrow_text_color)
            else:
                sns.scatterplot(data=dfpc, x=xlabel, y=ylabel, ax=figures["nocat"][axi])
            
                for k, feature in enumerate(_features):
                    #ax.plot([0,_loadings[k, 0] ], [0,_loadings[k, 1] ],color=arrow_color)
                    figures["nocat"][axi].arrow(0, 0, _loadings[k, 0],_loadings[k, 1],color=arrow_color,width=0.005,head_width=0.1)
                    figures["nocat"][axi].text(_loadings[k, 0],_loadings[k, 1],feature,color=arrow_text_color)
    
            dfpc_list.append(dfpc)
            combnum+=1
        
        if len(category)!=0:
            for cat in category:
                figures[cat]["fig"].suptitle(title)
                figures[cat]["fig"].tight_layout(pad=0.5)
                _save(save, cat+"_PCA", fig=figures[cat]["fig"])
        else:
            figures["nocat"]["fig"].suptitle(title)
            figures["nocat"]["fig"].tight_layout(pad=0.5)
            _save(save, "PCA")
        
        if explained_variance==True:
            fig, ax2=plt.subplots()
            exp_var_pca = pca.explained_variance_ratio_
            #
            # Cumulative sum of eigenvalues; This will be used to create step plot
            # for visualizing the variance explained by each principal component.
            #
            cum_sum_eigenvalues = np.cumsum(exp_var_pca)
            #
            # Create the visualization plot
            #
            xlabel=["pc"+str(i+1) for i in range(0,len(exp_var_pca))]
            plt.bar(xlabel, exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
            plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
            plt.ylabel('Explained variance ratio')
            plt.xlabel('Principal component index')
            _save(save, "ExplainedVar")
        else:
            ax2=None
        if show==True:
            plt.show()
        return {"data": dfpc_list,"pca": pca, "axes":figures, "axes_explained":ax2}
    elif method=="nmf":
        nmf=NMF(n_components=component,**nmfparam)
        if regularization:
            x=x/np.sum(x,axis=0)[None,:]
        W = nmf.fit_transform(x)
        H = nmf.components_
        combnum=0
        for axi, (i, j) in enumerate(comb):
            xlabel, ylabel='p'+str(i+1), 'p'+str(j+1)
            dfpc = pd.DataFrame(data = np.array([W[:,i],W[:,j]]).T, columns = [xlabel, ylabel],index=original_index)
            if len(category)!=0:
                for cat in category:
                    dfpc[cat]=df[cat]
                    if combnum==1:
                        sns.scatterplot(data=dfpc, x=xlabel, y=ylabel, hue=cat, ax=figures[cat]["axes"][axi])
                        figures[cat]["axes"][axi].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
                    else:
                        sns.scatterplot(data=dfpc, x=xlabel, y=ylabel, hue=cat, ax=figures[cat]["axes"][axi],
                                        legend=False)
            else:
                sns.scatterplot(data=dfpc, x=xlabel, y=ylabel, hue=category, ax=figures["nocat"][axi])
            dfpc_list.append(dfpc)
            combnum+=1
        if len(category)!=0:
            for cat in category:
                figures[cat]["fig"].suptitle(title)
                figures[cat]["fig"].tight_layout(pad=0.5)
                _save(save, cat+"_NMF", fig=figures[cat]["fig"])
        else:
            figures["nocat"]["fig"].suptitle(title)
            figures["nocat"]["fig"].tight_layout(pad=0.5)
        _save(save, "NMF")
        if explained_variance==True:
            fig, axes2=plt.subplots(nrows=component, figsize=[5,5])
            axes2=axes2.flatten()
            for i, ax in enumerate(axes2):
                if i==0:
                    ax.set_title("Coefficients of matrix H")
                ax.bar(np.arange(len(features)),H[i])
                ax.set_ylabel("p"+str(i+1))
                ax.set_xticks(np.arange(len(features)),labels=[])
            ax.set_xticks(np.arange(len(features)),labels=features, rotation=90)
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
            axes2=None
        if show==True:
            plt.show()
        return {"data": dfpc_list, "W":W, "H":H,"axes":figures,"axes_explained":axes2}
    elif method=="lda":
        lda=LatentDirichletAllocation(n_components=component, random_state=0)
        if regularization:
            x=x/np.sum(x,axis=0)[None,:]
        
    else:
        raise Exception('{} is not in options. Available options are: pca, nmf'.format(method))
def manifoldplot(df: pd.DataFrame,
                 variables: List=[],
                 category: Union[List, str]="", 
                 method: str="tsne",
                 show: bool=False,
                 figsize=[5,5],
                 title: str="",
                 param: dict={},
                 save: str="",ax: Optional[plt.Axes]=None,
                 **kwargs):
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
    method_dict={"random_projection": "Sparse random projection",
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
    x, category=_separate_data(df, variables=variables, category=category)
   
    x=zscore(x, axis=0)
    features=df.columns
    original_index=df.index
    embedding=_get_embedding(method=method,param=param)
    Xt=embedding.fit_transform(x)
    dft = pd.DataFrame(data = np.array([Xt[:,0],Xt[:,1]]).T, columns = ["d1", "d2"],index=original_index)
    
    if len(category) !=0:
        figsize=[5*len(category),5]
        fig, axes=plt.subplots(figsize=figsize, ncols=len(category))
        axes=axes.flatten()
        for cat,ax in zip(category, axes):
            dft[cat]=df[cat]
            sns.scatterplot(data=dft, x="d1", y="d2", hue=cat, ax=ax,**kwargs)
    else:
        fig, axes=plt.subplots(figsize=figsize)
        sns.scatterplot(data=dft, x="d1", y="d2", ax=axes,**kwargs)
    if title !="":
        fig.suptitle(title)
    else:
        fig.suptitle(method_dict[method])
    if show==True:
        plt.show()
    _save(save, method_dict[method])
    return {"data": dft, "axes": axes}

def clusterplot(df,
                variables: List=[],
                category: Union[List[str], str]="", 
                method: str="kmeans",
                n_clusters: Union[str , int]=3,
                x: str="",
                y: str="",
                size: float=10,
                reduce_dimension: str="umap", 
                testrange: list=[1,20],
                topn_cluster_num: int=2,
                show: bool=False,
                min_dist: float=0.25,
                n_neighbors: int=15,
                eps: Union[List[float], float]=0.5,
                pcacomponent: Optional[int]=None,
                ztranform: bool=True,
                palette=["Spectral","cubehelix"],
                save: str="",
                title: str="",
                barrierfree: bool=True,ax: Optional[plt.Axes]=None,
                piesize_scale: float=0.02,**kwargs)->Dict:
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
    size: float, optional (default: 10)
        The size of points in the scatter plot.
        
    testrange: list, optional (default: [1,20])
        The range of cluster numbers to be tested when n_clusters="auto".
    topn_cluster_num: int, optional (default: 2)
        Top n optimal cluster numbers to be plotted when n_clusters="auto".
    
    show: bool, optional (default: False)
        Whether to show figures
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
    
    
    original_index=df.index
    
    X, category=_separate_data(df, variables=variables, category=category)
    
    
    if ztranform:
        X=zscore(X, axis=0)
        
    if pcacomponent==None:
            
        if 20<X.shape[1]:
            pcacomponent=20
        elif 10<X.shape[1]:
            pcacomponent=10
        else:
            pcacomponent=2
    pca=PCA(n_components=pcacomponent, random_state=1)
    xpca=pca.fit_transform(X)
    
    if reduce_dimension=="umap":
        import umap
        u=umap.UMAP(random_state=42, min_dist=min_dist,n_neighbors=n_neighbors)
        X=u.fit_transform(xpca)
    
    if n_clusters=="auto" and method=="kmeans":
        Sum_of_squared_distances = []
        K = list(range(*testrange))
        for k in K:
            km = KMeans(n_clusters=k,n_init=10)
            km = km.fit(X)
            Sum_of_squared_distances.append(km.inertia_)
        normy=np.array(Sum_of_squared_distances)/np.amax(Sum_of_squared_distances)
        normy=1-normy
        normx=np.linspace(0,1, len(K))
        perp=_calc_curveture(normx, normy)
        # perp=[]
        # for i, (nx, ny) in enumerate(zip(normx, normy)):
        #     if i==0:
        #         perp.append(0)
        #         continue
        #     r=(nx**2+ny**2)**0.5
        #     sina=ny/r
        #     cosa=nx/r
        #     sinamb=sina*np.cos(np.pi*0.25)-cosa*np.sin(np.pi*0.25)
        #     perp.append(r*sinamb)
        # perp=np.array(perp)
        srtindex=np.argsort(perp)[::-1]
        plt.subplots()
        plt.plot(K, Sum_of_squared_distances, '-', label='Sum of squared distances')
        plt.plot(K, perp*np.amax(Sum_of_squared_distances), label="curveture")
        for i in range(topn_cluster_num):
            plt.plot([K[srtindex[i]],K[srtindex[i]]],[0,np.amax(Sum_of_squared_distances)], "--", color="r")
            plt.text(K[srtindex[i]], np.amax(Sum_of_squared_distances)*0.95, "N="+str(K[srtindex[i]]))
        plt.xticks(K)
        plt.xlabel('Cluster number')
        plt.ylabel('Sum of squared distances')
        plt.title('Elbow method for optimal cluster number')    
        plt.legend()
        #print("Top two optimal cluster No are: {}, {}".format(K[srtindex[0]],K[srtindex[1]]))
        #n_clusters=[K[srtindex[0]],K[srtindex[1]]]
        n_clusters=[ K[i] for i in srtindex[:topn_cluster_num]]
        print("Top two optimal cluster No are:", n_clusters)
        _save(save, method)
    elif n_clusters=="auto" and method=="fuzzy":
        try:
            import skfuzzy as fuzz
        except ImportError:
            from pip._internal import main as pip
            pip(['install', '--user', 'scikit-fuzzy'])
            import skfuzzy as fuzz
        fpcs = []
        K = list(range(*testrange))
        _X=X.T
        for nc in K:
            
            cntr, u, u0, d, jm, p, fpc = fuzz.cmeans(_X, nc, 2, error=0.005, maxiter=1000, init=None)
            
            fpcs.append(fpc)
        
        srtindex=np.argsort(fpcs)[::-1]
        plt.subplots()
        plt.plot(K, fpcs, '-')
     
        for i in range(topn_cluster_num):
            plt.plot([K[srtindex[i]],K[srtindex[i]]],[0,np.amax(fpcs)], "--", color="r")
            plt.text(K[srtindex[i]], np.amax(fpcs)*0.95, "N="+str(K[srtindex[i]]))
        plt.xticks(K)
        plt.xlabel('Cluster number')
        plt.ylabel('Fuzzy partition coefficient')
        n_clusters=[ K[i] for i in srtindex[:topn_cluster_num]]
        print("Top two optimal cluster No are:", n_clusters)
        
        
        _save(save, method)
    elif n_clusters=="auto" and method=="hierarchical":
        import scipy.spatial.distance as ssd
        
        labels=df.index
        D=ssd.squareform(ssd.pdist(xpca))
        Y = sch.linkage(D, method='ward')
        Z = sch.dendrogram(Y,labels=labels,no_plot=True)
        
        K = list(range(*testrange))
        newK=[]
        scores=[]
        for k in K:
            t=_dendrogram_threshold(Z, k)
            Z2=sch.dendrogram(Y,
                                labels = labels,
                                color_threshold=t,no_plot=True) 
            clusters=_get_cluster_classes(Z2, label='ivl')
            _k=len(clusters)
            if not _k in newK:
                newK.append(_k)
                sample2cluster={}
                i=1
                for k, v in clusters.items():
                    for sample in v:
                        sample2cluster[sample]="C"+str(i)
                    i+=1
                scores.append(silhouette_score(X, [sample2cluster[sample] for sample in labels], metric = 'euclidean')/_k)

        scores=np.array(scores)
        srtindex=np.argsort(scores)[::-1]
        plt.subplots()
        plt.plot(newK, scores, '-')
        for i in range(topn_cluster_num):
            plt.plot([newK[srtindex[i]],newK[srtindex[i]]],[0,np.amax(scores)], "--", color="r")
            plt.text(newK[srtindex[i]], np.amax(scores)*0.95, "N="+str(newK[srtindex[i]]))
        plt.xticks(newK)
        plt.xlabel('Cluster number')
        plt.ylabel('Silhouette scores')
        plt.title('Optimal cluster number searches by silhouette method')    
        
        n_clusters=[ newK[i] for i in srtindex[:topn_cluster_num]]
        print("Top two optimal cluster No are:", n_clusters)
        _save(save, method)
    elif n_clusters=="auto" and method=="dbscan":

        from sklearn.neighbors import NearestNeighbors
        neigh = NearestNeighbors(n_neighbors=2)
        nbrs = neigh.fit(X)
        distances, indices = nbrs.kneighbors(X)
        distances = np.sort(distances[:,1], axis=0)

        K=np.linspace(np.amin(distances), np.amax(distances),20)
        newK=[]
        scores=[]
        _K=[]
        for k in K:
            db = DBSCAN(eps=k, min_samples=5, n_jobs=-1)
            dbX=db.fit(X)
            labels=np.unique(dbX.labels_[dbX.labels_>=0])
  
            if len(labels)<2:
                continue
            _k=len(labels)
            if not _k in newK:
                newK.append(_k)
                _K.append(k)
                scores.append(silhouette_score(X[dbX.labels_>=0], dbX.labels_[dbX.labels_>=0], metric = 'euclidean')/_k)

        scores=np.array(scores)
        
        _ksort=np.argsort(newK)
        _K=np.array(_K)[_ksort]
        newK=np.array(newK)[_ksort]
        scores=np.array(scores)[_ksort]
        srtindex=np.argsort(scores)[::-1]
        plt.subplots()
        plt.plot(newK, scores, '-')
        
        for i in range(topn_cluster_num):
            plt.plot([newK[srtindex[i]],newK[srtindex[i]]],[0,np.amax(scores)], "--", color="r")
            plt.text(newK[srtindex[i]], np.amax(scores)*0.95, "N="+str(newK[srtindex[i]]))
        plt.xticks(newK)
        plt.xlabel('eps')
        plt.ylabel('Silhouette scores')
        plt.title('Optimal cluster number searches by silhouette method')    

        _n_clusters=[ newK[i] for i in range(topn_cluster_num)]
        print("Top two optimal cluster No are:", _n_clusters)
        eps=[_K[i] for i in srtindex[:topn_cluster_num]]
        _save(save, method)
    else:
        n_clusters=[n_clusters]
    if method=="kmeans":
        dfnews=[]
        if reduce_dimension=="umap":
            x="UMAP1"
            y="UMAP2"
        for nc in n_clusters:
            kmean = KMeans(n_clusters=nc, random_state=0,n_init=10)
            kmX=kmean.fit(X)
            labels=np.unique(kmX.labels_)
            
            dfnew=pd.DataFrame(data = np.array([X[:,0],X[:,1]]).T, columns = [x, y], index=original_index)
            dfnew["kmeans"]=kmX.labels_
            dfnews.append(dfnew)
        hue="kmeans"
        
    elif method=="hierarchical":
        import scipy.spatial.distance as ssd
        labels=df.index
        D=ssd.squareform(ssd.pdist(xpca))
        Y = sch.linkage(D, method='ward')
        Z = sch.dendrogram(Y,labels=labels,no_plot=True)
        if reduce_dimension=="umap":
            x="UMAP1"
            y="UMAP2"
        dfnews=[]
        for nc in n_clusters:
            t=_dendrogram_threshold(Z, nc)
            Z2=sch.dendrogram(Y,
                                labels = labels,
                                color_threshold=t,no_plot=True) 
            clusters=_get_cluster_classes(Z2, label='ivl')
            sample2cluster={}
            i=1
            for k, v in clusters.items():
                for sample in v:
                    sample2cluster[sample]="C"+str(i)
                i+=1
                
            dfnew=pd.DataFrame(data = np.array([X[:,0],X[:,1]]).T, columns = [x, y], index=original_index)
            dfnew["hierarchical"]=[sample2cluster[sample] for sample in labels]       
            dfnews.append(dfnew)
        hue="hierarchical"
    elif method=="dbscan":
        dfnews=[]
        if reduce_dimension=="umap":
            x="UMAP1"
            y="UMAP2"
        if type(eps)==float:
            eps=[eps]
        n_clusters=[]
        for e in eps:
            db = DBSCAN(eps=e, min_samples=5, n_jobs=-1)
            dbX=db.fit(X)
            labels=np.unique(dbX.labels_)
            
            dfnew=pd.DataFrame(data = np.array([X[:,0],X[:,1]]).T, columns = [x, y], index=original_index)
            dfnew["dbscan"]=dbX.labels_
            dfnews.append(dfnew)
            tmp=0
            for c in set(dbX.labels_):
                if c >=0:
                    tmp+=1
            n_clusters.append(str(tmp)+", eps="+str(np.round(e,2)))
            
            
        hue="dbscan"
    elif method=="hdbscan":
        dfnews=[]
        if reduce_dimension=="umap":
            x="UMAP1"
            y="UMAP2"
        
        try:
            import hdbscan
        except ImportError:
            from pip._internal import main as pip
            pip(['install', '--user', 'hdbscan'])
            import hdbscan

        if type(eps)==float:
            eps=[eps]
        n_clusters=[]
        fuzzylabels=[]
        for e in eps:
            db = hdbscan.HDBSCAN(min_cluster_size=10, 
                                 prediction_data=True,
                                 algorithm='best', 
                                 alpha=1.0, 
                                 approx_min_span_tree=True,
                                gen_min_span_tree=True, leaf_size=40,
                                metric='euclidean', min_samples=None, p=None)
            dbX=db.fit(X)
            labels=np.unique(dbX.labels_)
            
            dfnew=pd.DataFrame(data = np.array([X[:,0],X[:,1]]).T, columns = [x, y], index=original_index)
            dfnew["dbscan"]=dbX.labels_
            fuzzylabels.append(hdbscan.all_points_membership_vectors(dbX))
            dfnews.append(dfnew)
            tmp=0
            for c in set(dbX.labels_):
                if c >=0:
                    tmp+=1
            n_clusters.append(str(tmp)+", eps="+str(np.round(e,2)))
            
            
        hue="hdbscan"
    elif method=="fuzzy":
        try:
            import skfuzzy as fuzz
        except ImportError:
            from pip._internal import main as pip
            pip(['install', '--user', 'scikit-fuzzy'])
            import skfuzzy as fuzz
        
        dfnews=[]
        fuzzylabels=[]
        if reduce_dimension=="umap":
            x="UMAP1"
            y="UMAP2"
        _X=X.T
        for nc in n_clusters:
            
            cntr, u, u0, d, jm, p, fpc = fuzz.cmeans(_X, nc, 2, error=0.005, maxiter=1000, init=None)
            
            dfnew=pd.DataFrame(data = np.array([X[:,0],X[:,1]]).T, columns = [x, y], index=original_index)
            fuzzylabels.append(u.T)
            dfnews.append(dfnew)
        hue="fuzzy"
        
    _dfnews={}
    
    if method=="fuzzy" or method=="hdbscan":
        for dfnew, K, fl in zip(dfnews, n_clusters, fuzzylabels): 
            if len(category)==0:
                fig, ax=plt.subplots(ncols=2, figsize=[8,4])
                ax=[ax]
            else:
                fig, ax=plt.subplots(ncols=2+len(category), figsize=[8+4*len(category),4])
            
            if type(K)==str:
                _K=K
                K, eps=_K.split(", ")
                K=int(K)
                _cmap=plt.get_cmap(palette[0], K)
            else:
                _cmap=plt.get_cmap(palette[0], K)
            colors=[]
            color_entropy=[]
            for c in fl:
                tmp=np.zeros([3])
                for i in range(K):
                    #print(_cmap(i))
                    #print(c[i])
                    tmp+=np.array(_cmap(i))[:3]*c[i]
                tmp=np.where(tmp>1, 1, tmp)
                colors.append(tmp)
                color_entropy.append(np.sum(tmp*np.log2(tmp+0.000001)))
                
            entropy_srt=np.argsort(color_entropy)
            colors=np.array(colors)[entropy_srt]
            ax[0].scatter(dfnew[x].values[entropy_srt], dfnew[y].values[entropy_srt], c=colors, s=size)
            #sns.scatterplot(data=dfnew,x=x,y=y,hue=hue, ax=ax[0], palette=palette[0],**kwargs)
            if method=="fuzzy":
                _title="Fuzzy c-means. Cluster num="+str(K)
            elif method=="hdbscan":
                _title="HDBSCAN. Cluster num="+_K
            ax[0].set_title(_title, alpha=0.5)
            legend_elements = [Line2D([0], [0], marker='o', color='lavender', 
                                      label="fuzzy"+str(i),
                                      markerfacecolor=_cmap(i), 
                                      markersize=10)
                      for i in range(K)]
    
            ax[0].legend(handles=legend_elements,loc="best")
            for i in range(K):
                dfnew[method+str(i)]=fl[:,i]
            
            pie_scatter(dfnew, x=x,y=y, 
                        category=[method+str(i) for i in range(K)],
                        piesize_scale=piesize_scale, 
                        ax=ax[1],
                        label="",bbox_to_anchor="best", title="Probability is represented by pie charts")
            
            
            if len(category)!=0:
                for i, cat in enumerate(category):
                    dfnew[cat]=df[cat]
                    sns.scatterplot(data=dfnew,x=x,y=y,hue=cat, ax=ax[i+2], palette=palette[1], s=size,**kwargs)
            _dfnews[K]=dfnew 
    else:
    
        for dfnew, K in zip(dfnews, n_clusters): 
            if len(category)==0:
                axnum=1
                fig, ax=plt.subplots(ncols=1, figsize=[4,4])
                ax=[ax]
            else:
                fig, ax=plt.subplots(ncols=1+len(category), figsize=[4+4*len(category),4])
            sns.scatterplot(data=dfnew,x=x,y=y,hue=hue, ax=ax[0], palette=palette[0], s=size,**kwargs)
            ax[0].set_title(method+" Cluster number="+str(K))
            if len(category)!=0:
                for i, cat in enumerate(category):
                    dfnew[cat]=df[cat]
                    sns.scatterplot(data=dfnew,x=x,y=y,hue=cat, ax=ax[i+1], palette=palette[1], s=size,**kwargs)
            _dfnews[K]=dfnew
    _save(save, method+"_scatter")
    return {"data": _dfnews, "axes":ax}

def volcanoplot():
    pass


def regression_single(df, 
                      x: str="",
                      y: str="", 
                      method: str="ransac",
                      category: str="", 
                      figsize: List[int]=[5,5],
                      show=False, ransac_param={"max_trials":1000},
                      robust_param={},
                      xunit: str="",
                      yunit: str="",
                      title: str="",
                      random_state: int=42,ax: Optional[plt.Axes]=None,
                      save: str="") -> Dict:
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
    
    
    Y=df[y]
    _X=np.array(df[x]).reshape([-1,1])
    X=np.array(df[x])
    plotline_X = np.arange(X.min(), X.max()).reshape(-1, 1)
    n = X.shape[0]
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title)
    plt.subplots_adjust(left=0.15)
    if method=="ransac":
        from sklearn.linear_model import RANSACRegressor
        
        
        
        fit_df=pd.DataFrame()
        fitted_model = RANSACRegressor(random_state=random_state,**ransac_param).fit(_X,Y)
        fit_df["ransac_regression"] = fitted_model.predict(plotline_X)
        coef = fitted_model.estimator_.coef_[0]
        intercept=fitted_model.estimator_.intercept_
        inlier_mask = fitted_model.inlier_mask_
        outlier_mask = ~inlier_mask
        
                                # number of samples
        y_model=fitted_model.predict(_X)

        r2 = _calc_r2(X,Y)
        # mean squared error
        MSE = 1/n * np.sum( (Y - y_model)**2 )
        
        # to plot the adjusted model
        x_line = plotline_X.flatten()
        y_line = fit_df["ransac_regression"]
         
        ci, pi, std_error=_ci_pi(X,Y,plotline_X.flatten(),y_model)
        q=((X-X.mean()).transpose() @ (X-X.mean()))
        sigma=std_error*(q**-1)**(0.5)
        coef_p=stats.t.sf(abs(fitted_model.estimator_.coef_[0]/sigma), df=X.shape[0]-2)
        ############### Ploting

        _draw_ci_pi(ax, ci, pi,x_line, y_line)
        sns.scatterplot(x=X[inlier_mask], y=Y[inlier_mask], color="blue", label="Inliers")
        sns.scatterplot(x=X[outlier_mask], y=Y[outlier_mask], color="red", label="Outliers")
        plt.xlabel(x)
        plt.ylabel(y)
        #print(r2, MSE,ransac_coef,ransac.estimator_.intercept_)
        plt.title("RANSAC regression, r2: {:.2f}, MSE: {:.2f}\ny = {:.2f} + {:.2f}x, coefficient p-value: {:.2E}".format(
            r2, MSE,coef,intercept,coef_p
            )
        )
        plt.plot(plotline_X.flatten(),fit_df["ransac_regression"])
        
        _save(save, "ransac")
        if len(category)!=0:
            fig, ax=plt.subplots(figsize=figsize)
            plt.subplots_adjust(left=0.15)
            _draw_ci_pi(ax, ci, pi,x_line, y_line)
            sns.scatterplot(data=df,x=x, y=y, hue=category)
            
            plt.xlabel(x)
            plt.ylabel(y)
            #print(r2, MSE,ransac_coef,ransac.estimator_.intercept_)
            plt.title("RANSAC regression, r2: {:.2f}, MSE: {:.2f}\ny = {:.2f} + {:.2f}x, coefficient p-value: {:.2E}".format(
                r2, MSE,coef,intercept,coef_p
                )
            )
            plt.plot(plotline_X.flatten(),fit_df["ransac_regression"])
            _save(save, "ransac_"+category)
    elif method=="robust":
        import statsmodels.api as sm
        rlm_model = sm.RLM(Y, sm.add_constant(X),
        M=sm.robust.norms.HuberT(),**robust_param)
        fitted_model = rlm_model.fit()
        summary=fitted_model.summary()
        coef=fitted_model.params[1]
        intercept=fitted_model.params[0]
        intercept_p=fitted_model.pvalues[0]
        coef_p=fitted_model.pvalues[1]
        y_model=fitted_model.predict(sm.add_constant(X))
        r2 = _calc_r2(X,Y)
        x_line = plotline_X.flatten()
        y_line = fitted_model.predict(sm.add_constant(x_line))
        
        ci, pi,std_error=_ci_pi(X,Y,plotline_X.flatten(),y_model)
        MSE = 1/n * np.sum( (Y - y_model)**2 )

        _draw_ci_pi(ax, ci, pi,x_line, y_line)
        sns.scatterplot(data=df,x=x, y=y, color="blue")
        #print(r2, MSE,ransac_coef,ransac.estimator_.intercept_)
        plt.title("Robust linear regression, r2: {:.2f}, MSE: {:.2f}\ny = {:.2f} + {:.2f}x , p-values: coefficient {:.2f}, \
        intercept {:.2f}".format(
            r2, MSE,coef,intercept,coef_p,intercept_p
            )
        )
        plt.plot(plotline_X.flatten(),y_line)
        _save(save, "robust")
        if len(category)!=0:
            fig, ax=plt.subplots(figsize=figsize)
            plt.subplots_adjust(left=0.15)
            _draw_ci_pi(ax, ci, pi,x_line, y_line)
            sns.scatterplot(data=df,x=x, y=y, hue=category)
            #print(r2, MSE,ransac_coef,ransac.estimator_.intercept_)
            plt.title("Robust linear regression, r2: {:.2f}, MSE: {:.2f}\ny = {:.2f} + {:.2f}x , p-values: coefficient {:.2f}, \
            intercept {:.2f}".format(
                r2, MSE,coef,intercept,coef_p,intercept_p
                )
            )
            plt.plot(plotline_X.flatten(),y_line)
            _save(save, "robust_"+category)
    elif method=="lasso" or method=="elastic_net" or method=="ols":
        if method=="lasso":
            method="sqrt_lasso"
        import statsmodels.api as sm
        rlm_model = sm.OLS(Y, sm.add_constant(X))
        if method=="ols":
            fitted_model = rlm_model.fit()
        else:
            fitted_model = rlm_model.fit_regularized(method)
        coef=fitted_model.params[1]
        intercept=fitted_model.params[0]
        y_model=fitted_model.predict(sm.add_constant(X))
        r2 = _calc_r2(X,Y)
        x_line = plotline_X.flatten()
        y_line = fitted_model.predict(sm.add_constant(x_line))
        ci, pi, std_error=_ci_pi(X,Y,plotline_X.flatten(),y_model)
        q=((X-X.mean()).transpose() @ (X-X.mean()))
        sigma=std_error*(q**-1)**(0.5)
        print(sigma,coef )
        coef_p=stats.t.sf(abs(coef/sigma), df=X.shape[0]-2)
        MSE = 1/n * np.sum( (Y - y_model)**2 )

        _draw_ci_pi(ax, ci, pi,x_line, y_line)   
        sns.scatterplot(data=df,x=x, y=y, color="blue")
        #print(r2, MSE,ransac_coef,ransac.estimator_.intercept_)
        plt.title("OLS ({}), r2: {:.2f}, MSE: {:.2f}\ny = {:.2f} + {:.2f}x, coefficient p-value: {:.2E}".format(method,
            r2, MSE,coef,intercept,coef_p
            )
        )
        plt.plot(plotline_X.flatten(),y_line)
        _save(save, method)
        if len(category)!=0:
            fig, ax=plt.subplots(figsize=figsize)
            plt.subplots_adjust(left=0.15)
            _draw_ci_pi(ax, ci, pi,x_line, y_line)
            sns.scatterplot(data=df,x=x, y=y, color="blue",hue=category)
            #print(r2, MSE,ransac_coef,ransac.estimator_.intercept_)
            plt.title("OLS ({}), r2: {:.2f}, MSE: {:.2f}\ny = {:.2f} + {:.2f}x, coefficient p-value: {:.2E}".format(method,
                r2, MSE,coef,intercept,coef_p
                )
            )
            plt.plot(plotline_X.flatten(),y_line)
            _save(save, method+"_"+category)
    return {"axes":ax, "coefficient":coef,"intercept":intercept,"coefficient_pval":coef_p, "r2":r2, "fitted_model":fitted_model}

def pie_scatter(df: pd.DataFrame,  
                x: str, 
                y: str, 
                category: list, 
                
                logscalex: bool=False,
                logscaley: bool=False,
                pie_palette: str="tab20c",
                label: Union[List, str]="all",topn=10,
                ax: Optional[plt.Axes]=None,
                piesizes: Union[List, str]="",
                save: str="",
                show: bool=False,
                edge_color: str="gray",
                min_piesize: float=0.3,
                figsize=[6,6],
                xunit: str="",
                yunit: str="",
                xlabel: str="",
                ylabel: str="", 
                title: str="",
                
                bbox_to_anchor: Union[List, str]=[0.95, 1],
                piesize_scale: float=0.01) -> dict:
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
    if type(pie_palette)== str:
        colors={}
        unique_labels=category
            
        cmap=plt.get_cmap(pie_palette)
        labelnum=len(unique_labels)
        for i, ul in enumerate(unique_labels):
            colors[ul]=cmap(i/labelnum)
    elif type(pie_palette)==dict:
        colors=pie_palette
        unique_labels=colors.keys()
    else:
        raise Exception("Unknown pie_palette type.")
    if ax ==None:
        fig, ax=plt.subplots(figsize=figsize)
    plt.subplots_adjust(right=0.80)
    X=df[x]
    Y=df[y]
    yscale=""
    xscale=""
    if logscaley==True:
        Y=np.log10(Y+1)
        yscale=" (scaled by log10)"
    if logscalex==True:
        X=np.log10(X+1)
        xscale=" (scaled by log10)"
    Frac=df[category]
    
    index=df.index
    piesize_scale=np.amax([np.amax(X), np.amax(Y)])*piesize_scale
    
    if piesizes=="sum_of_each":
        sums=Frac.sum(axis=1)
        sumsrt=np.argsort(sums)[::-1]
        sumsrt=set(sumsrt[:topn])
        sums=sums/np.amax(sums)
        sums=piesize_scale*(sums+min_piesize)
    _colors=[colors[f] for f in unique_labels]
    for i, (_x, _y, _ind) in enumerate(zip(X, Y, index)):
        _frac=Frac.loc[_ind].values 
        _frac=2*np.pi*np.array(_frac)/np.sum(_frac)
        
        angle=0
        #print(sums.loc[_ind])
        for fr, co in zip(_frac, _colors):
            if type(piesizes)==str:
                if piesizes=="sum_of_each":
                    _baumkuchen_xy(ax, _x, _y, angle, fr, 0, sums.loc[_ind],20, co, edge_color=edge_color)
                elif piesizes=="":
                    _baumkuchen_xy(ax, _x, _y, angle, fr, 0, piesize_scale,20, co, edge_color=edge_color)
                else:
                    pass
            elif type(piesizes)==list and len(piesizes) !=0:
                _baumkuchen_xy(ax, _x, _y, angle, fr, 0, piesize_scale*piesizes[i],20, co, edge_color=edge_color)
            else:
                _baumkuchen_xy(ax, _x, _y, angle, fr, 0, piesize_scale,20, co, edge_color=edge_color)
            angle+=fr
        
        if type(label)==str:
            if label=="all":
                ax.text(_x, _y,_ind)
            elif label=="topn_of_sum":
                if i in sumsrt:
                    ax.text(_x, _y,_ind)
                
            elif label=="":
                pass
        elif type(label)==list:
            if _ind in label:
                ax.text(_x, _y,_ind)
            
            
    if xlabel!="":
        x=xlabel
    if ylabel!="":
        y=ylabel
    plt.xlabel(x+xscale)
    plt.ylabel(y+yscale)
    legend_elements = [Line2D([0], [0], marker='o', color='lavender', label=ul,markerfacecolor=colors[ul], markersize=10)
                      for ul in unique_labels]
    if type(bbox_to_anchor)==str:
        ax.legend(handles=legend_elements,loc=bbox_to_anchor)
    else:
        ax.legend(handles=legend_elements,bbox_to_anchor=bbox_to_anchor)
    ax.set_title(title)
    if yunit!="":
        ax.text(0, 1, "({})".format(yunit), transform=ax.transAxes, ha="right")
    if xunit!="":
        ax.text(1, 0, "({})".format(xunit), transform=ax.transAxes, ha="left",va="top")
    _save(save, "pie_scatter")
    return {"axes":ax}



if __name__=="__main__":
    
    
    
    #test="dotplot"
    #test="triangle_heatmap"
    test="decomp"
    test="manifold"
    test="triangle_heatmap"
    test="radialtree"
    test="violinplot"
    test="cluster"
    test="regression"
    test="complex_clustermap"
    test="dotplot"
    test="regression"
    test="nice_piechart_num"
    test="pie_scatter"
    test="correlation"
    test="manifold"
    test="stacked"
    test="stackedlines"
    test="correlation"
    test="stacked"
    if test=="stackedlines":
        f="/media/koh/grasnas/home/data/omniplot/energy/owid-energy-data.csv"
        df=pd.read_csv(f)
        _df=df.loc[df["country"]=="Japan"]
        cols=['biofuel_consumption',
             'coal_consumption',
             'gas_consumption',
             'hydro_consumption',
             'nuclear_consumption',
             'oil_consumption',
             'other_renewable_consumption',
             'solar_consumption',
             'wind_consumption']
        stackedlines(df=_df, x="year",y=cols,title="Japan", remove_all_zero=True, inverse=True,show_values=True, yunit="twh")
        
        _df=pd.DataFrame({"x":np.arange(100),
                          "y0":np.random.normal(loc=0.0, scale=1.0, size=100)-3,
                          "y1":np.random.normal(loc=0.0, scale=1.0, size=100)+3,
                          "y2":np.random.normal(loc=0.0, scale=1.0, size=100)-4})
        stackedlines(df=_df, x="x",y=["y0","y1", "y2"],title="Japan",bbox_to_anchor=[1,1], sort=True, remove_all_zero=False, inverse=False,show_values=False, yunit="twh")
        
        plt.show()
    elif test=="correlation":
        df=sns.load_dataset("penguins")
        df=df.dropna(axis=0)
        
            
        correlation(df, category=["species", "island","sex"], method="pearson", ztransform=True)
        plt.show()
    elif test=="nice_piechart":
        df=sns.load_dataset("penguins")
        df=df.dropna(axis=0)
        tmp=[]
        for num, (sp, i ,se) in enumerate(zip(df["species"], df["island"],df["sex"])):
            if num/df.shape[0] > 0.98:
                tmp.append("NA")
            else:
                tmp.append(sp[0]+"_"+i[0]+"_"+se[0])
        df["combine"]=tmp
        print(df)
        nice_piechart(df, category=["species", "island","sex","species", "island","sex","combine"],ncols=4)
        plt.show()
    elif test=="regression":
        df=sns.load_dataset("penguins")
        df=df.dropna(axis=0)
        regression_single(df, x="bill_length_mm",y="body_mass_g", method="robust",category="species")
        plt.show()
    elif test=="dotplot":
        df=pd.DataFrame({"Experiments":["exp1","exp1","exp1","exp1","exp2","exp2","exp3"],
                         "GO":["nucleus","cytoplasm","chromosome","DNA binding","chromosome","RNA binding","RNA binding"],
                         "FDR":[10,1,5,3,1,2,0.5],
                         "odds":[3.3,1.1,2.5,2.1,0.8,2.3,0.9]})
        dotplot(df, row="GO",col="Experiments", size_val="FDR",color_val="odds", highlight="FDR",
        color_title="Odds", size_title="-log10 p",scaling=20)
        # df=pd.read_csv("/home/koh/ews/idr_revision/clustering_analysis/cellloc_longform.csv")
        # print(df.columns)
        # df=df.fillna(0)
        # #dotplot(df, size_val="pval",color_val="odds", highlight="FDR",color_title="Odds ratio", size_title="-log10 p value",scaling=20)
        #
        # dotplot(df, row="Condensate",col="Cluster", size_val="pval",color_val="odds", highlight="FDR",
        #         color_title="Odds", size_title="-log10 p value",scaling=20)
        plt.show()
    elif test=="triangle_heatmap":
        s=20
        mat=np.arange(s*s).reshape([s,s])
        import string, random
        letters = string.ascii_letters+string.digits
        labels=[''.join(random.choice(letters) for i in range(10)) for _ in range(s)]
        df=pd.DataFrame(data=mat, index=labels, columns=labels)
        triangle_heatmap(df,grid_pos=[2*s//10,5*s//10,7*s//10],grid_labels=["A","B","C","D"])
    elif test=="complex_clustermap":
        # _df=pd.DataFrame(np.arange(100).reshape([10,10]))
        # cmap=plt.get_cmap("tab20b")
        # complex_clustermap(_df,
        #                    row_colormap={"test1":[cmap(v) for v in np.linspace(0,1,10)]},
        #                    row_plot={"sine":np.sin(np.linspace(0,np.pi,10))},
        #                    approx_clusternum=3,
        #                    merginalsum=True)
        df=sns.load_dataset("penguins")
        
        df=df.dropna(axis=0)
        dfcol=pd.DataFrame({"features":["bill","bill","flipper"]})
        complex_clustermap(df,
                           dfcol=dfcol,
                            variables=["bill_length_mm","bill_depth_mm","flipper_length_mm"],
                            row_colors=["species","sex"],
                            row_scatter=["body_mass_g"],
                            row_plot=["body_mass_g"],
                            row_bar=["body_mass_g"],
                            col_colors=["features"],
                            approx_clusternum=3,
                            merginalsum=True, title="Penguins")
        plt.show()
    elif test=="radialtree":
        df=sns.load_dataset("penguins")
        df=df.dropna(axis=0)
        variables=["bill_length_mm","bill_depth_mm","flipper_length_mm"]
        radialtree(df, variables=variables, category=["species","island","sex"])
        plt.show()
    elif test=="decomp":
        df=sns.load_dataset("penguins")
        df=df.dropna(axis=0)
        variables=["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]

        decomplot(df, variables=variables,category=["species","sex"],method="pca")
        plt.show()
    elif test=="manifold":
        df=sns.load_dataset("penguins")
        df=df.dropna(axis=0)
        variables=["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]
        #df=df[features]
        manifoldplot(df, 
                     variables=variables,
                     category=["species", "island"],
                     method="tsne")
        plt.show()
    elif test=="cluster":
        df=sns.load_dataset("penguins")
        df=df.dropna(axis=0)
        features=["species","sex","bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]
        df=df[features]
        #clusterplot(df,category=["species","sex"],method="hierarchical",n_clusters="auto")
        #clusterplot(df,category=["species","sex"],method="fuzzy",n_clusters="auto", piesize_scale=0.03,topn_cluster_num=3)
        #clusterplot(df,category=["species","sex"],method="hdbscan",eps=0.35)
        clusterplot(df,category=["species","sex"],method="hierarchical",n_clusters="auto")
        plt.show()
    elif test=="violinplot":
        df=sns.load_dataset("penguins")
        df=df.dropna(axis=0)
        violinplot(df,x="species",y="bill_length_mm", 
                   pairs=[["Adelie","Chinstrap" ],["Gentoo","Chinstrap" ],["Adelie","Gentoo" ]],
                   test="mannwhitneyu",
                   significance="symbol",swarm=True)
        plt.show()
    elif test=="stacked": 
        df=sns.load_dataset("penguins")
        df=df.dropna(axis=0)
        stacked_barplot(df, x=["species","island"],
                         hue=["sex","island"], scale="absolute", order=[
                                                                        ["Chinstrap","Gentoo","Adelie"],
                                                                        ["Dream","Biscoe","Torgersen"]],
                         test_pairs=[["Adelie","Gentoo"]], hatch=True)
        stacked_barplot(df, x="species",
                         hue="sex", scale="percentage", hatch=True)
        plt.show()
    elif test=="pie_scatter":
        f="/home/koh/ews/omniplot/data/energy_vs_gdp.csv"
        df=pd.read_csv(f, comment='#')
        df=df.set_index("country")
        pie_scatter(df, x="gdppc",y="pop", category=['biofuel_electricity',
                                                     'coal_electricity',
                                                     'gas_electricity',
                                                     'hydro_electricity',
                                                     'nuclear_electricity',
                                                     'oil_electricity',
                                                     'other_renewable_electricity',
                                                     'solar_electricity',
                                                     'wind_electricity'],logscalex=True,logscaley=True,
                                                    sizes="sum_of_each",
                                                    min_piesize=0.1,piesize=0.05, label="topn_of_sum")
        
        
        plt.show()
    
    elif test=="nice_piechart_num":
        f="/home/koh/ews/omniplot/data/energy_vs_gdp.csv"
        df=pd.read_csv(f)
        df=df.set_index("country", comment='#')
        nice_piechart_num(df, hue=['biofuel_electricity',
                                                     'coal_electricity',
                                                     'gas_electricity',
                                                     'hydro_electricity',
                                                     'nuclear_electricity',
                                                     'oil_electricity',
                                                     'other_renewable_electricity',
                                                     'solar_electricity',
                                                     'wind_electricity'],ncols=10)
        
        
        plt.show()