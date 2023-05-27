"""
Heatmap functions
"""
import matplotlib.collections as mc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import cm
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import leaves_list
from joblib import Parallel, delayed
# from collections import defaultdict
import matplotlib.colors
from natsort import natsort_keygen, natsorted
import scipy.cluster.hierarchy as sch
import fastcluster as fcl
# from decimal import Decimal
# import sys 
import matplotlib as mpl
from sklearn.cluster import KMeans #, DBSCAN
# from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
# from sklearn.decomposition import PCA, NMF, LatentDirichletAllocation
# from scipy.stats import fisher_exact
from scipy.stats import zscore
# from itertools import combinations
# import os
#script_dir = os.path.dirname( __file__ )
#sys.path.append( script_dir )
# import scipy.stats as stats
# import itertools as it
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle,Ellipse, Polygon #, Circle, RegularPolygon
import copy
import textwrap
from matplotlib.artist import Artist
# from matplotlib.transforms import Affine2D
# import mpl_toolkits.axisartist.floating_axes as floating_axes
import matplotlib.patheffects as patheffects
from typing import Union, Optional, Dict, List

from omniplot.utils import _separate_data, _save, _create_color_markerlut, _get_cluster_classes2, _dendrogram_threshold, _get_cluster_classes, _separate_cdata #
from omniplot.utils import colormap_list, shape_list, marker_list

__all__=["correlation", "triangle_heatmap", "complex_clustermap","dotplot", "heatmap"]
def correlation(df: pd.DataFrame, 
                category: Union[str, list]=[],
                variables: List=[],
                method="pearson",
                palette: str="coolwarm",
                figsize: List=[6,6],
                show_values: bool=False,
                clustermap_param: dict={},
                ztransform: bool=True,
                xticklabels: bool=False,
                yticklabels: bool=False,
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
    if ztransform is True:
        X=zscore(X, axis=0)
    if method=="pearson":
        dmat=np.corrcoef(X)
    else:
        dmat=squareform(pdist(X, method))
    if method=="pearson":
        ctitle="Pearson correlation"
    else:
        ctitle=method+" distance"    

    dfm=pd.DataFrame(data=dmat, columns=original_index,index=original_index)


    colnames=dfm.columns

        
    if len(category) >0:
        for cat in category:
            dfm[cat]=df[cat].values
        res=complex_clustermap(dfm,
                               heatmap_col=colnames, 
                               row_colors=category,
                               ztranform=False,
                               xticklabels=xticklabels,
                               yticklabels=yticklabels,
                               figsize=figsize,
                               ctitle=ctitle,**clustermap_param)
    else:
        
        res=complex_clustermap(dfm,
                         heatmap_col=colnames, 
                         xticklabels=xticklabels,
                         yticklabels=yticklabels,
                   method="ward", 
                   heatmap_palette=palette,
                   col_cluster=True,
                   row_cluster=True,
                   figsize=figsize,
                   ctitle=ctitle,
                   ztranform=False,
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
                     save: str="",
                     title: str="")-> dict:
    
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
    cmap = plt.get_cmap("Reds")
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
                       approx_clusternum: int=3,
                       approx_clusternum_col: int=3,
                       color_var: int=0,
                       marginalsum: bool=False,
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
                       treepalette: str="tab20b",
                       **kwargs):
    """
    Drawing a clustered heatmap with marginal plots.
    
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
    marginalsum : bool, optional
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
    """
    #print(kwargs)
    rnum, cnum=df.shape
    if len(heatmap_col)==0 and len(variables)==0:
        raise ValueError("Please specify the variables option")
    if len(heatmap_col)!=0:
        variables=heatmap_col
    cnum=len(variables)
    if len(figsize)==0:
        xsize=np.amin([cnum, 20])
        xsize=np.amax([xsize, 5])
        figsize=[xsize,10]
    scatterpointsize=5
    sns.set(font_scale=1)
    original=df[variables].values
    if ztranform==True:
        df[variables]=df[variables].apply(zscore)
        if ctitle =="":
            ctitle="zscore"
    
    if len(col_plot)!=0 or len(col_scatter)!=0 or len(col_bar)!=0:
        if dfcol==None:
            raise ValueError("if you want to plot along the x axis, you need provide dfcol option containing values to plot.")
    cdata={"Cluster":[],"Index":[],"RGB":[]}
    totalrowplot=0
    if marginalsum==True:
        totalrowplot+=1
    totalrowplot+=len(row_plot)
    totalrowplot+=len(row_colors)
    totalrowplot+=len(row_scatter)
    totalrowplot+=len(row_bar)
    totalcolplot=0
    if marginalsum==True:
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
        
        if marginalsum:
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
        if marginalsum:
            mat=original
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
    _cmap=plt.get_cmap(treepalette)
    if color_var>0:
        
        cmap = _cmap(np.linspace(0, 1, color_var))
    else:
        cmap = _cmap(np.linspace(0, 1, approx_clusternum+5))
    sch.set_link_color_palette([mpl.colors.rgb2hex(rgb[:3]) for rgb in cmap])
    
    """coloring the row dendrogram based on branch numbers crossed with the threshold"""
    
    if g.dendrogram_row != None:
        t=_dendrogram_threshold(g.dendrogram_row.dendrogram,approx_clusternum)
               
        den=sch.dendrogram(g.dendrogram_row.linkage,
                            labels = g.data.index,
                            color_threshold=t,
                            ax=g.ax_row_dendrogram,
                            orientation="left")
        
        # print(np.amax(den["dcoord"]))
        g.ax_row_dendrogram.invert_yaxis()
        clusters = _get_cluster_classes(den)
        
        keys=list(clusters.keys())
        # print(keys)
        ckeys={}
        i=1
        for k in keys:
            if k.startswith("C0_"):
                ckeys[k]=k
            else:
                ckeys[k]="C"+str(i)
                i+=1
        for c, v in clusters.items():
            
            _c=ckeys[c]
            if c.startswith("C0_"):
                c="#000000"
            for _v in v:
                cdata["Cluster"].append(_c)
                cdata["Index"].append(_v)
                cdata["RGB"].append(matplotlib.colors.to_rgb(c))
        """Setting the row dendrogram ends here"""
    
    
    """coloring the col dendrogram based on branch numbers crossed with the threshold"""
    col_cdata={"Cluster":[],"Index":[],"RGB":[]}
    if g.dendrogram_col != None:
        t=_dendrogram_threshold(g.dendrogram_col.dendrogram,approx_clusternum_col)
        den=sch.dendrogram(g.dendrogram_col.linkage,
                                                 labels = g.data.columns,
                                                 color_threshold=t,ax=g.ax_col_dendrogram,
                            orientation="top")  
        #g.ax_col_dendrogram.invert_yaxis()
        col_clusters = _get_cluster_classes(den)
        col_cdata={"Cluster":[],"Index":[],"RGB":[]}
        keys=list(col_clusters.keys())
        col_ckeys={}
        i=1
        for k in keys:
            if k.startswith("C0_"):
                col_ckeys[k]=k
            else:
                col_ckeys[k]="C"+str(i)
                i+=1
        for c, v in col_clusters.items():
            
            _c=col_ckeys[c]
            if c.startswith("C0_"):
                c="#000000"
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




def heatmap(df: pd.DataFrame,
                variables: list=[],
                category: list=[],

                row: str="",
                col: str="",
                fillna: float=0,
                colors: str="",
                
                dtype: str="numerical",
                column_wise_color: bool=False,

                sizes: Union[str, list, np.ndarray, pd.DataFrame]="",
                size_title: str="",

                clustering_method: str="hierarchical",
                kmeans_kw: dict={},

                method: str="ward",
                metric: str="euclidean",
                row_cluster: bool=True,
                row_split: bool=False,
                col_cluster: bool=True,
                col_split: bool=False,

                shape: str="rectangle",
                shape_colors: Union[np.ndarray, list, dict]=[],
                edgecolor: str="w",
                
                rowlabels: str="",
                row_ticklabels: bool=True,
                rowrot: int=0,
                col_ticklabels: bool=True,
                colrot: int=90,
                
                palette:str="coolwarm",
                cpalette:str="tab20b",
                row_colors: Union[dict, list]={},
                col_colors: Union[dict, list]={},
                row_plot: Union[dict, list]={},
                col_plot: Union[dict, list]={},
                plotkw: dict={},
                row_scatter: Union[dict, list]={},
                col_scatter: Union[dict, list]={},
                scatterkw: dict={},
                row_bar: Union[dict, list]={},
                col_bar: Union[dict, list]={},
                barkw: dict={},
                row_axis: int=0,
                col_axis: int=0,


                approx_clusternum: Union[int, str]=3,
                approx_clusternum_col: int=3,

                cunit: str="",
                show: bool=False,
                ztranform: bool=True,

                figsize: list=[],
                title: str="",
                cbar_title: str="",

                save: str="",
                treepalette: str="tab20b",
                above_threshold_color="black",
                rasterized: bool=True,
                size_format: str="",
                rowplot_format: str="{x:.1f}",
                colplot_format:  str="{x:.1f}",
                
                boxlabels: bool=False,
                box_textwidth: Optional[int]=None,
                box_max_lines: Optional[int]=None,
                n_jobs: int=-1,
                show_values: bool=False,
                text_color: str="w",
                val_format: str="",
                )->Dict:
    """
    Drawing a heatmap. The function is mostly overlapping with the complex_clustermap, but has more flexibility, but may be slower.
    The main difference is this heatmap uses patch collections instead of pcolormech.  
    
    Parameters
    ----------
    df : pandas DataFrame
        The default assumes the data is a wide form. If the data is a long form, pass 'row' and 'col' options to specify the heatmap rows and columns, and 
        'colors' option for the heatmap color. Otherwise, you may specify variables and category options. 

    variables: list, optional
        The column names of variables for heatmap. 
        If not set, all columns except ones specified by the 'category' will be used for the heatmap.
    
    categry: list, optional
        The column names of variables for categorical colors (for rows)

    rowlabels: str, optional
        The column names of row labels. If not set, index will be used as rowlabels. 

    row: str, optional
        The column name of the heatmap row. Only applied when the data is a long form
    col: str,optional
        The column name of the heatmap col. Only applied when the data is a long form
    fillna: float, optional (default: 0)
        Filling nan values with this value
    
    colors: str, optional
        The column name of the heatmap color value. Only applied when the data is a long form

    sizes: Union[str, list, np.ndarray, pd.DataFrame], optional
        Values to be proportional to each heatmap element. For a wide form data, it accepts only a matrix-like data 
        (list or np.ndarray or dataframe) with the same shape of the df option. For a long form data, a string of the 
        column name of size values can be accepted.
    
    size_title: str, optional
        The title of sizes

    clustering_method: str, optional (default: "hierarchical")
        This option is not implemented yet
    
    method: str, optional (defaul: "ward") ["single", "ward","average","weighted","centroid","median"]
        The method for the linkage calculation
    
    metric: str, optional (default: "euclidean")
        The metric for the distance calculation

    row_cluster: bool, optional (default: True)
        Whether to perform the clustering analysis by rows
    
    row_split: bool, optional (default: False)
        Whether to split the heatmap according to the clusters

    col_cluster: bool, optional (default: True)
        Whether to perform the clustering analysis by columns
    
    col_split: bool, optional (default: False)
        Whether to split the heatmap according to the clusters

    shape: str="rectangle", ["rectangle", "circle", "triangle", "star", "polygon:n", "star:n:m", "by_category"]
        The shape of the heatmap elements. For polygon:n, n is an integer of vertices of the polygon (e.g., polygon:5 will be a pentagon).
        Likewise, star:n:m can specify star's vertices. 'star:5:2' is a classical star shape and equivalent to 'star'. 'by_category' will 
        produce a variety of shapes to represent categorical values.

    edgecolor: str, optional (default: "w")
        The edgecolor of the heatmap elements
    
    row_ticklabels: bool, optional (default: True)
        Whether to show row tick labels.
    
    rowrot: int, optional (default: 0)
        The angle of row labels

    col_ticklabels: bool, optional (default: True)
        Whether to show column tick labels.

    colrot: int, optional (default: 90)
        The angle of column labels
    
    palette: str="coolwarm"
        The colormap name for the heatmap

    col_colors: dictionary, optional
        A dictionary for column-wise color labels. The number of elemens for each value must be equal to 
        the number of the df columns.  
    
    row_plot, col_plot : list, optional
        Dictionaries for the values to be plotted as lines.
    plotkw: dict, optional
        the keyword arguments for pyplot.plot

    row_scatter, col_scatter: list, optional
        Dictionaries for the values to be plotted as points.
    scatterkw: dict, optional
        the keyword arguments for pyplot.scatter
    
    row_bar, col_bar: list, optional
        Dictionaries for the values to be plotted as bars.
    barkw: dict, optional
        the keyword arguments for pyplot.bar

    approx_clusternum : int, optional (default: 3)
        The approximate number of row clusters to be created. Labeling the groups of leaves with different colors. 
        The result of hierarchical clustering won't change.    
    approx_clusternum_col : int, optional (default: 3)
        The approximate number of column clusters to be created. Labeling the groups of leaves with different colors. 
        The result of hierarchical clustering won't change.

    show : bool, optional
        Whether or not to show the figure.
    cunit: str, optional
        The color bar unit
    cbar_title: str, optional
        The color bar title
    show: bool, optional (default: False)
        Whether to show the figure

    ztranform: bool, optional (default: True)
        Whether to ztransform values

    figsize: list, optional
        the figure size. 

    title: str, optional
        The title of the figure.

    save: str, optional
        The prefix of a saving file name.

    treepalette: str, optional (default: "tab20b")
        The colormap of dendrograms
    
    above_threshold_color, optional (default: "black")
        The color of dendrogram branches above the threshold

    rasterized: bool, optional (default: False)
        Whether to rasterize the figure
    
    size_format: str, optional
        The format of size values in the legend

    rowplot_format: str, optional (default: "{x:.1f}")
        The format of y axis values of row plots/scatter/bar

    colplot_format:  str, optional (default: "{x:.1f}")
        The format of y axis values of row plots/scatter/bar
    
    boxlabels: bool, optional (default: False)
        Whether to add row labels boxed by clustering groups. 
    
    box_textwidth: int, optional
        The number of charcters in one line of the boxlabels
    
    box_max_lines: Optional[int]=None,
        The maximum number of lines in the boxlabels.
    
    n_jobs: int, optional (default: -1)
        The number of threads to use for adding pathes/shapes to the heatmap.

    show_values: bool, optional (default: False)
        Whether to show values over the heatmap elements.

    text_color: str="w",
        The color of texts for value annotation.

    val_format: str="",
        The format of texts for value annotation. e.g., {x:.2f}

    Returns
    -------
        {"row_clsuter":rclusters,"col_cluster":cclusters, "axes":axis_dict, "colsort":sortindexc,"rowsort":sortindexr} : dict
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

    ax_dict={}
    row_colors=copy.deepcopy(row_colors)
    col_colors=copy.deepcopy(col_colors)
    
    if col_split==True and row_split==True:
        raise Exception("Splitting both columns and rows may not be a good idea...")
    
    def _scale_size(x, size_scale, smin, smax):
        return size_scale*((x-smin)/(smax-smin))
    def _reverse_size(x, size_scale, smin, smax):
        return (x/size_scale)*(smax-smin)+smin
    
    margin=0.00
    sns.set_theme(style="white",font="Arial",font_scale=1.1)
    TBLR=['top','bottom','left','right']
    # Separating values into heatmap colors, sizes, ans category    
    lut={}
    if dtype=="numerical":
        (df, X, category, rowlabels, collabels, 
        Xsize, size_title, cunit, lut)=_process_vdata(df, variables, category,
                                                        row,  col,rowlabels, 
                                                        sizes, size_title, 
                                                        colors, lut,fillna,ztranform,cunit)
    elif  dtype=="categorical":
        (df, X, category, rowlabels, collabels, 
         Xsize, size_title, cunit, lut)=_process_cdata(df, variables,category ,
                                                       row,  col,rowlabels, sizes, size_title, 
                                                       colors, lut,fillna,cunit)
        if len(shape_colors)>0:
            if type(shape_colors)==dict:
                scX=np.array(list(shape_colors.values())[0])
            else:
                scX=np.array(shape_colors)
        if row_cluster==True or col_cluster==True:
            clustering_method="kmodes"
            print("Categorical variables are provided. 'kmodes' will be used for the clustering method.")
    
    rowplot_num=len(category)+len(row_colors)+\
        len(row_plot)+len(row_scatter)+len(row_bar)+row_axis+\
        int(clustering_method=="kmeans" and row_cluster==True)+\
        int(clustering_method=="kmodes" and row_cluster==True)
    colplot_num=len(col_colors)+len(col_plot)+\
        len(col_scatter)+len(col_bar)+col_axis+\
        int(clustering_method=="kmeans" and col_cluster==True)+\
        int(clustering_method=="kmodes" and col_cluster==True)
    Xshape=X.shape
    if np.sum(Xshape)>200:
        edgecolor=None
    show_legend=int(rowplot_num>0 or type(Xsize)!=type(None))

    if len(figsize)==0:
        figure_height=np.amin([np.amax([X.shape[0]/5, 3]), 10])
        if show_legend>0:
            figure_width=figure_height*X.shape[1]/X.shape[0]+1
            figure_width=np.amin([np.amax([figure_height*X.shape[1]/X.shape[0]+1+int(boxlabels)*6, 5]), 10])
        else:
            figure_width=figure_height*X.shape[1]/X.shape[0]+int(boxlabels)*6
        figsize=[figure_width,figure_height,]
    print("figsize: ", figsize)
    fig=plt.figure(figsize=figsize, layout='constrained')
    
    # determining the size and positionn of axes
    (xori, yori, 
    boxwidth, 
    legendw, legendh,
    ltreew, ltreeh,  
    ttreey, ttreew,  ttreeh, 
    hmapx, hmapw, hmaph, 
    lcatx, lcatw, 
    tcaty, tcath, 
    row_ticklabels)=_axis_loc_sizes(col_ticklabels, row_ticklabels,
                    boxlabels, 
                    row_cluster,col_cluster,
                    show_legend,
                    rowplot_num,colplot_num,
                    collabels,
                    Xshape)
    
    size_legend_num=3
    size_legend_elements=[]
    if type(Xsize)!=type(None):
        size_legend_elements, size_labels =_create_shape_legend_elements(Xsize, 
                                                                        Xshape,
                                                                        hmapw,
                                                                        hmaph,
                                                                        legendw,
                                                                        legendh,
                                                                        _scale_size,
                                                                        _reverse_size,
                                                                        size_legend_num,
                                                                        shape,
                                                                        size_format)

    # Row-wise clustering 
    legend_elements_dict={} 
    rclusters={}
    cclusters={}
    if row_cluster==True:
        if clustering_method=="hierarchical":
            # Drawing the left dendrogram
            ax0=fig.add_axes([xori,yori,ltreew,ltreeh])
            X, Xsize, Z, rowlabels, rclusters=_dendrogram(X, 
                                                                    Xsize, ax0,rowlabels,
                                                                    treepalette ,
                                                                    approx_clusternum, 
                                                                    metric,method,above_threshold_color, "left")

            ax0.axis('off')
            ax0.margins(y=margin)
            sortindexr=Z["leaves"]
            ax_dict["lefttree"]=ax0
            if len(shape_colors)>0:
                scX=scX[sortindexr]
        elif clustering_method=="kmeans":
            _kmeans = KMeans(n_clusters=approx_clusternum, random_state=0, n_init="auto").fit(X, *kmeans_kw)
            row_colors["kmeans_row"]=_kmeans.labels_.astype(str)
            sortindexr=np.lexsort((X.mean(axis=1),_kmeans.labels_))
            X=X[sortindexr]
            if type(Xsize)!=type(None):
                Xsize=Xsize[sortindexr]
            if len(shape_colors)>0:
                scX=scX[sortindexr]
            rowlabels=rowlabels[sortindexr]
            _klabels=_kmeans.labels_[sortindexr]
            for i, (kclass, label) in enumerate(zip(_klabels, rowlabels)):
                if not kclass in rclusters:
                    rclusters[kclass]=[]
                rclusters[kclass].append(label)
        elif clustering_method=="kmodes":
            try:
                from kmodes.kmodes import KModes
            except ImportError:
                from pip._internal import main as pip
                pip(['install', '--user', 'kmodes'])
                from kmodes.kmodes import KModes
            km = KModes(n_clusters=3, init='Huang', n_init=5, verbose=1)
            clusters = km.fit_predict(X)
            row_colors["kmodes_row"]=clusters.astype(str)
            sortindexr=np.argsort(clusters)
            X=X[sortindexr]
            if type(Xsize)!=type(None):
                Xsize=Xsize[sortindexr]

            if len(shape_colors)>0:
                scX=scX[sortindexr]
            rowlabels=rowlabels[sortindexr]
            _klabels=clusters[sortindexr]
            for i, (kclass, label) in enumerate(zip(_klabels, rowlabels)):
                if not kclass in rclusters:
                    rclusters[kclass]=[]
                rclusters[kclass].append(label)

    else:
        sortindexr=np.arange(len(rowlabels))

    # Column-wise clustering 
    if col_cluster==True:
        
        if clustering_method=="hierarchical":
            ax1=fig.add_axes([hmapx,ttreey,ttreew,ttreeh])
            X, Xsize, Zt, collabels, cclusters,=_dendrogram(X, 
                                                            Xsize, 
                                                            ax1,
                                                            collabels,
                                                            treepalette ,
                                                            approx_clusternum_col, 
                                                            metric,
                                                            method,
                                                            above_threshold_color, 
                                                            "top")     
            ax1.axis('off')
            ax1.margins(x=margin)
            sortindexc=Zt["leaves"]
            if len(shape_colors)>0:
                scX=scX[:, sortindexc]
            ax_dict["toptree"]=ax1
        elif clustering_method=="kmeans":
            _ckmeans = KMeans(n_clusters=approx_clusternum, random_state=0, n_init="auto").fit(X.T, *kmeans_kw)
            col_colors["kmeans_column"]=_ckmeans.labels_.astype(str)
            sortindexc=np.lexsort((X.mean(axis=0),_ckmeans.labels_))
            X=X[:,sortindexc]
            if type(Xsize)!=type(None):
                Xsize=Xsize[:, sortindexc]

            if len(shape_colors)>0:
                scX=scX[:, sortindexc]

            collabels=collabels[sortindexc]
            _cklabels=_ckmeans.labels_[sortindexc]
            for i, (kclass, label) in enumerate(zip(_cklabels, collabels)):
                if not kclass in cclusters:
                    cclusters[kclass]=[]
                cclusters[kclass].append(label)
        elif clustering_method=="kmodes":
            try:
                from kmodes.kmodes import KModes
            except ImportError:
                from pip._internal import main as pip
                pip(['install', '--user', 'kmodes'])
                from kmodes.kmodes import KModes
            km = KModes(n_clusters=3, init='Huang', n_init=5, verbose=1)
            clustersc = km.fit_predict(X.T)
            col_colors["kmodes_column"]=clustersc.astype(str)
            sortindexc=np.argsort(clustersc)
            X=X[:, sortindexc]
            if type(Xsize)!=type(None):
                Xsize=Xsize[:, sortindexc]

            if len(shape_colors)>0:
                scX=scX[:, sortindexc]
            collabels=collabels[sortindexc]
            _cklabels=clustersc[sortindexc]
            # print(_cklabels,collabels )
            for i, (kclass, label) in enumerate(zip(_cklabels, collabels)):
                if not kclass in cclusters:
                    cclusters[kclass]=[]
                cclusters[kclass].append(label)
    else:
        sortindexc=np.arange(len(collabels))
    catnum=0
    legend_elements_dict, catnum, axis_dict=_row_plot(df, 
                                           sortindexr, 
                                           fig, 
                                           category, 
                                           lut,
                                           row_colors,
                                           row_plot, 
                                           row_scatter, 
                                           row_bar, 
                                           rowplot_format, 
                                           legend_elements_dict, 
                                           Xshape, 
                                           lcatx,lcatw,
                                           yori,hmaph, margin, 
                                           plotkw, scatterkw, barkw, catnum, ax_dict,row_axis=row_axis)
    legend_elements_dict, catnum, axis_dict=_col_plot(sortindexc, 
                                                      fig, 
                                                      col_colors, 
                                                      col_plot, 
                                            col_scatter, 
                                            col_bar, 
                                            colplot_format,
                                            legend_elements_dict, 
                                            Xshape, 
                                            tcaty,
                                            tcath,
                                            hmapx,
                                            hmapw, 
                                            margin, 
                                            plotkw, scatterkw, barkw, catnum, axis_dict, col_axis=col_axis)
    # Creating the heatmap
    # Calculating the maximum and minum values of the data if the data is numerical.
    cmap=plt.get_cmap(palette)

    if dtype=="numerical":
        Xmin=np.amin(X)
        Xmax=np.amax(X)
        if val_format=="":
            if 1<np.abs(Xmax)<=1000:
                val_format="{x:.2f}"
            elif 0<np.abs(Xmax)<=1 or 1000<np.abs(Xmax):
                val_format="{x:.3E}"


    if len(shape_colors)>0:
        scXmin=np.amin(scX)
        scXmax=np.amax(scX)
    if type(Xsize)!=type(None):
        Xsizemin=np.amin(Xsize)
        Xsizemax=np.amax(Xsize)

    # Creating the color legend lists if the data is categorical
    shape_legend_elements={}
    if dtype=="categorical":
        _shape_lut={}
        _color_lut={}
        if column_wise_color==False:
            _uniq_labels=np.unique(X.flatten())
            if shape=="by_category":
                
                _shape_lut={u: shape_list[i] for i, u in enumerate(_uniq_labels)}
                _shape_legend_elements={"ao":[],"labels":[],"handler":{}}
                for _cat, _shape in _shape_lut.items():
                    ao=_AnyObject()
                    obshape=_AnyObjectHandler(facecolor="black", shape=_shape)
                    _shape_legend_elements["ao"].append(ao)
                    _shape_legend_elements["labels"].append(_cat)
                    _shape_legend_elements["handeler"][ao]=obshape
                shape_legend_elements["categorical"]=_shape_legend_elements
            else:
                
                _cmap=plt.get_cmap(cpalette, len(_uniq_labels)+2)
                _color_lut={u: _cmap(i+1) for i, u in enumerate(_uniq_labels)}
                
                legend_elements=[]
                for _cat, _color in _color_lut.items():
                    legend_elements.append(Line2D([0], [0], marker="s", linewidth=0, markeredgecolor="darkgray",
                                        label=_cat,
                                        markerfacecolor=_color))
                legend_elements_dict["categorical"]=legend_elements
        else:
            for i in range(Xshape[1]):
                __X=X[:,i]
                _uniq_labels=np.unique(__X)
                if shape=="by_category":
                    _shape_legend_elements={"ao":[],"labels":[],"handler":{}}
                    _shape_tmp_lut={u: shape_list[i] for i, u in enumerate(_uniq_labels)}
                    for _cat, _shape in _shape_tmp_lut.items():
                        ao=_AnyObject()
                        obshape=_AnyObjectHandler(facecolor="black", shape=_shape)
                        _shape_legend_elements["ao"].append(ao)
                        _shape_legend_elements["labels"].append(_cat)
                        _shape_legend_elements["handler"][ao]=obshape
                    shape_legend_elements[collabels[i]]=_shape_legend_elements
                    _shape_lut.update(_shape_tmp_lut)
                else:
                    _cmap=plt.get_cmap(colormap_list[i+int(row_cluster)+int(col_cluster)], len(_uniq_labels)+2)
                    _tmp_lut={u: _cmap(i+1) for i, u in enumerate(_uniq_labels)}
                    
                    legend_elements=[]
                    for _cat, _color in _tmp_lut.items():
                        legend_elements.append(Line2D([0], [0], marker="s", linewidth=0, markeredgecolor="darkgray",
                                            label=_cat,
                                            markerfacecolor=_color))
                    legend_elements_dict[collabels[i]]=legend_elements
                    _color_lut.update(_tmp_lut)
                

    if shape=="by_category":
        shape=_shape_lut
    #the number of members in row clusters
    if row_cluster==True:
        if clustering_method=="hierarchical":
            _cnums_dict, ccolor_unique=_get_cluster_classes2(Z,above_threshold_color)
            # print("_cnums_dict", "ccolor_unique", _cnums_dict, ccolor_unique)
            cnums=np.array([_cnums_dict[k] for k in ccolor_unique])
        else:
            u, i, c=np.unique(_klabels, return_counts=True, return_index=True
                    )
            cnums=c[np.argsort(i)]
        _cnums=cnums/np.sum(cnums)

    #the number of members in column clusters
    if col_cluster==True:
        if clustering_method=="hierarchical":
            _col_cnums_dict, col_ccolor_unique=_get_cluster_classes2(Zt,above_threshold_color)
            # print("_cnums_dict", "ccolor_unique", _cnums_dict, ccolor_unique)
            col_cnums=np.array([_col_cnums_dict[k] for k in col_ccolor_unique])
        else:
            u, i, c=np.unique(_cklabels, return_counts=True, return_index=True
                    )
            col_cnums=c[np.argsort(i)]
        _col_cnums=col_cnums/np.sum(col_cnums)
    
    #Boxed row labels by clusters
    render = fig.canvas.get_renderer()
    rwidth=render.width
    if row_cluster==True and boxlabels==True:
        rownum=len("".join(rowlabels))
        if box_textwidth==None:
            box_textwidth=20
            linenum=rownum//box_textwidth+int(rownum%box_textwidth!=0)
        if box_max_lines==None:
            box_max_lines=np.amin([10,linenum])
        wrapper = textwrap.TextWrapper(width=20, max_lines=box_max_lines)
        _r=0
        _y=0
        boxfont=12
        ax=fig.add_axes([hmapx+hmapw,yori+_y,boxwidth,hmaph])
        ax.axis('off')
        for i, (_c,c) in enumerate(zip(_cnums, cnums)):
            text=", ".join(rowlabels[_r:_r+c])
            word_list = wrapper.wrap(text=text)
            # print(word_list)
            t=ax.text(0.1,_y+_c-0.01,"\n".join(word_list), fontsize=boxfont,va="top",
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1, alpha=0.8))
            bb=t.get_window_extent(renderer=render)
            # print("width", rwidth, bb.width)
            if i==0 and bb.width/rwidth> boxwidth:
                while bb.width/rwidth > boxwidth:
                    Artist.remove(t)
                    boxfont-=0.5
                    # print(boxfont)
                    if boxfont<0:
                        break
                    t=ax.text(0.1,_y+_c-0.01,"\n".join(word_list), fontsize=boxfont, va="top",
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1, alpha=0.8))
                    bb=t.get_window_extent(renderer=render)


            _r+=c
            _y+=_c

    #Drawing a heatmap
    pcolormesh=False
    if Xshape[0]>=512 or Xshape[1]>=512:
        pcolormesh=True
    if row_split==True and row_cluster==True:
        
        # print(cnums, _cnums)
        _y=0
        _r=0
        for i, (_c,c) in enumerate(zip(_cnums, cnums)):
            #print([hmapx,yori+_y,hmapw,hmaph*_c])
            ax=fig.add_axes([hmapx,yori+_y,hmapw,hmaph*_c])
            Xsub=X[_r:_r+c]
            Xsubshape=Xsub.shape
            
            if pcolormesh==True:
                if dtype=="numerical":
                    Xsub=(Xsub-Xmin)/(Xmax-Xmin)
                    ax.pcolormesh(Xsub)
                elif dtype=="categorical":
                    ax.pcolormesh(
                        np.array([_color_lut[u] 
                    for u in Xsub.flatten()]
                    ).reshape([Xsubshape[0],Xsubshape[1], -1]))
            else:
                _X=Xsub.flatten()
                if dtype=="numerical":
                    _X=(_X-Xmin)/(Xmax-Xmin)
                    facecolors=cmap(_X)
                elif dtype=="categorical":
                    if type(shape)==dict:
                        if len(shape_colors)>0:
                            Xsub=scX[_r:_r+c]
                            Xsubshape=scX.shape
                            _scX=Xsub.flatten()
                            _scX=(_scX-scXmin)/(scXmax-scXmin)
                            facecolors=cmap(_scX)
                        else:
                            facecolors=["black" for u in _X]
                    else:
                        facecolors=[_color_lut[u] for u in _X]
                _Xsize=None
                if type(Xsize)!=type(None):
                    _Xsize=Xsize[_r:_r+c]
                    _Xsize=(_Xsize-Xsizemin)/(Xsizemax-Xsizemin)
                    _Xsize=_Xsize.flatten()
                row_col=[ [j,i] for i in range(Xsub.shape[0]) for j in range(Xsub.shape[1])]
                
                _add_patches(facecolors, 
                            Xsubshape, 
                            shape, 
                            row_col, 
                            None, 
                            Xsize, 
                            _Xsize,
                            ax,
                            row_ticklabels,
                            rowlabels[_r:_r+c], 
                            rowrot, 
                            col_ticklabels, 
                            collabels, 
                            colrot,rasterized, n_jobs=n_jobs)
                if show_values==True:
                    for (_xtmp, _ytmp), text in zip(row_col, Xsub.flatten()):
                        txt=ax.text(_xtmp, _ytmp, val_format.format(x=text), color=text_color, ha="center",va="center")
                        txt.set_path_effects([patheffects.withStroke(linewidth=1, foreground='black')])


                ax.margins(x=margin,y=margin)
            ax.tick_params(pad=1,axis='both', which='both', length=0)
            
            if _y>0:
                ax.set_xticks([])
                
            _r+=c
            _y+=hmaph*_c
            if c==1:
                for pos in TBLR:
                    ax.spines[pos].set_color('gray')
            else:
                for pos in TBLR:
                    ax.spines[pos].set_color('black')

            axis_dict["heatmap"+str(i)]=ax

    elif col_split==True and col_cluster==True:
        
        # print(cnums, _cnums)
        _y=0
        _r=0
        for i, (_c,c) in enumerate(zip(_col_cnums, col_cnums)):
            #print([hmapx,yori+_y,hmapw,hmaph*_c])
            ax=fig.add_axes([hmapx+_y,yori,hmapw*_c,hmaph])
            
            Xsub=X[:,_r:_r+c]
            Xsubshape=Xsub.shape
            if pcolormesh==True:
                if dtype=="numerical":
                    Xsub=(Xsub-Xmin)/(Xmax-Xmin)
                    ax.pcolormesh(Xsub)
                elif dtype=="categorical":
                    ax.pcolormesh(
                        np.array([_color_lut[u] 
                    for u in Xsub.flatten()]
                    ).reshape([Xsubshape[0],Xsubshape[1], -1]))
            else:
                _X=Xsub.flatten()
                if dtype=="numerical":
                    _X=(_X-Xmin)/(Xmax-Xmin)
                    facecolors=cmap(_X)
                elif dtype=="categorical":
                    if type(shape)==dict:
                        if len(shape_colors)>0:
                            Xsub=scX[_r:_r+c]
                            Xsubshape=scX.shape
                            _scX=Xsub.flatten()
                            _scX=(_scX-scXmin)/(scXmax-scXmin)
                            facecolors=cmap(_scX)
                        else:
                            facecolors=["black" for u in _X]
                    else:
                        facecolors=[_color_lut[u] for u in _X]
                _Xsize=None
                if type(Xsize)!=type(None):
                    Xsizesub=Xsize[:, _r:_r+c]
                    _Xsize=Xsizesub.flatten()
                    _Xsize=(_Xsize-Xsizemin)/(Xsizemax-Xsizemin)
                row_col=[ [j,i] for i in range(Xsub.shape[0]) for j in range(Xsub.shape[1])]
                
                _add_patches(facecolors, 
                            Xsubshape, 
                            shape, 
                            row_col, 
                            None, 
                            Xsize, 
                            _Xsize,
                            ax,
                            row_ticklabels,
                            rowlabels, 
                            rowrot, 
                            col_ticklabels, 
                            collabels[_r:_r+c], 
                            colrot,rasterized, n_jobs=n_jobs)
                if show_values==True:
                    for (_xtmp, _ytmp), text in zip(row_col, Xsub.flatten()):
                        txt=ax.text(_xtmp, _ytmp, val_format.format(x=text), color=text_color, ha="center",va="center")
                        txt.set_path_effects([patheffects.withStroke(linewidth=1, foreground='black')])

                ax.margins(x=margin,y=margin)
            ax.tick_params(pad=1,axis='both', which='both', length=0)
            if i<len(col_cnums)-1:
                ax.set_yticks([])
            
            _r+=c
            _y+=hmapw*_c
            if c==1:
                for pos in TBLR:
                    ax.spines[pos].set_color('gray')
            else:
                for pos in TBLR:
                    ax.spines[pos].set_color('black')

            axis_dict["heatmap"+str(i)]=ax
    else:
        ax=fig.add_axes([hmapx,yori,hmapw,hmaph])

        if pcolormesh==True:
            if dtype=="numerical":
                X=(X-Xmin)/(Xmax-Xmin)
                ax.pcolormesh(X)
            elif dtype=="categorical":
                ax.pcolormesh(
                    np.array([_color_lut[u] 
                for u in X.flatten()]
                ).reshape([Xshape[0],Xshape[1], -1]))
        else:

            _X=X.flatten()
            _Xsize=None
            if type(Xsize)!=type(None):
                _Xsize=Xsize.flatten()
                _Xsize=(_Xsize-Xsizemin)/(Xsizemax-Xsizemin)
            if dtype=="numerical":
                _X=(_X-Xmin)/(Xmax-Xmin)
                facecolors=cmap(_X)
            elif dtype=="categorical":
                if type(shape)==dict:
                    if len(shape_colors)>0:

                        _scX=scX.flatten()
                        _scX=(_scX-scXmin)/(scXmax-scXmin)
                        facecolors=cmap(_scX)
                    else:
                        facecolors=["black" for u in _X]
                else:
                    facecolors=[_color_lut[u] for u in _X]
                
            row_col=[ [j,i] for i in range(X.shape[0]) for j in range(X.shape[1])]
            
            _add_patches(facecolors, Xshape, shape, row_col, edgecolor, Xsize, 
                        _Xsize,ax,row_ticklabels,rowlabels, rowrot, 
                        col_ticklabels, collabels, colrot,rasterized,Xflatten=_X, n_jobs=n_jobs)
            
            if show_values==True:
                for (_xtmp, _ytmp), text in zip(row_col, X.flatten()):
                    txt=ax.text(_xtmp, _ytmp, val_format.format(x=text), color=text_color, ha="center",va="center")
                    txt.set_path_effects([patheffects.withStroke(linewidth=1, foreground='black')])

            ax.margins(x=margin,y=margin)
        ax.tick_params(pad=1,axis='both', which='both', length=0)
        axis_dict["heatmap"]=ax
    
    #Setting categorical color legends
    legendnum=0
    if len(legend_elements_dict) >0:
        for legendnum, (cat, legend_elements) in enumerate(legend_elements_dict.items()):
            axlegend=fig.add_axes([hmapx+hmapw+boxwidth+0.01,ttreey-legendnum*0.15,legendw,legendh])
            axlegend.add_artist(axlegend.legend(handles=legend_elements, 
                                                title=cat,bbox_to_anchor=(0.0,0.5),
                                                loc="center left"))
            axlegend.axis('off')
            axis_dict["legend_"+cat]=axlegend

    if row_axis>0 or col_axis >0:
        for i in range(row_axis):
            legendnum+=1            
            axlegend=fig.add_axes([hmapx+hmapw+boxwidth+0.01,ttreey-legendnum*0.15,legendw,legendh])
            axlegend.axis('off')
            axis_dict["row_legend"+str(i)]=axlegend
        for i in range(col_axis):
            legendnum+=1            
            axlegend=fig.add_axes([hmapx+hmapw+boxwidth+0.01,ttreey-legendnum*0.15,legendw,legendh])
            axlegend.axis('off')
            axis_dict["col_legend"+str(i)]=axlegend


    if len(shape_legend_elements) >0:
        for cat, legend_elements in shape_legend_elements.items():
            legendnum+=1

            axlegend=fig.add_axes([hmapx+hmapw+boxwidth+0.01,ttreey-legendnum*0.15,legendw,legendh])
            axlegend.add_artist(axlegend.legend(legend_elements["ao"], 
            legend_elements["labels"],
            handler_map=legend_elements["handler"],
            title=cat,
            bbox_to_anchor=(0.0,0.5),
            loc="center left"))
            axlegend.axis('off')
            axis_dict["legend_"+cat]=axlegend

    #Setting the size legend
    if type(Xsize)!=type(None):
        if legendnum>0:
            legendnum+=1
        axlegend=fig.add_axes([hmapx+hmapw+0.02,0.14,legendw,legendh])
        _pc = PatchCollection(size_legend_elements, edgecolor=["white"]+["darkgray"]*size_legend_num,
                              facecolor=["white"]+["darkgray"]*size_legend_num, 
                              alpha=[0,0.8,0.8,0.8])
        for _i, (_label, pos) in enumerate(size_labels):
            axlegend.text(0.1, pos, _label, color="black", va="center")
        axlegend.add_collection(_pc)

        axlegend.axis('off')

        axlegend.margins(0.0)
        if size_title!="":
            axlegend.set_title(size_title)
        axlegend.set_facecolor('lavender')
        axis_dict["size"]=axlegend


    # Setting the color bar
    if dtype=="numerical":
        axc=fig.add_axes([hmapx+hmapw+0.02,0.09,0.15,0.02])
        norm = mpl.colors.Normalize(vmin=np.amin(X), vmax=np.amax(X))
        cb1 = mpl.colorbar.ColorbarBase(axc, cmap=cmap,
                                        norm=norm,
                                        orientation='horizontal')
        if cbar_title!="":
            axc.set_title(cbar_title)
        if ztranform==True:
            cb1.set_label('z-score')
        elif cunit!="":
            cb1.set_label(cunit)
        axis_dict["colorbar"]=axc
    elif len(shape_colors)>0:
        axc=fig.add_axes([hmapx+hmapw+0.02,0.09,0.15,0.02])
        norm = mpl.colors.Normalize(vmin=np.amin(scX), vmax=np.amax(scX))
        cb1 = mpl.colorbar.ColorbarBase(axc, cmap=cmap,
                                        norm=norm,
                                        orientation='horizontal')
        if cbar_title!="":
            axc.set_title(cbar_title)
        if cunit!="":
            cb1.set_label(cunit)
        axis_dict["colorbar"]=axc

    fig.suptitle(title)
    # plt.subplots_adjust(**gridspec_kw)
    _save(save, "heatmap")
    if show==True:
        plt.show()

    return {"row_clsuter":rclusters,"col_cluster":cclusters, "axes":axis_dict, "colsort":sortindexc,"rowsort":sortindexr}

def _add_patches(facecolors: Union[List, np.ndarray], 
                 Xshape: Union[List, tuple, np.ndarray], 
                 shape: Union[str, dict], 
                 row_col: list, 
                 edgecolor, 
                 Xsize, 
                 _Xsize,
                 ax: plt.Axes,
                 row_ticklabels,
                 rowlabels, 
                 rowrot, 
                 col_ticklabels, 
                 collabels, 
                 colrot,
                 rasterized,
                 Xflatten=None,
                 n_jobs=-1):
    
    if type(Xsize)!=type(None):
        _row_col=np.array(row_col)
        _shapes = [Rectangle((- 0.5,- 0.5), 
                             np.amax(_row_col[:,0])+1, 
                             np.amax(_row_col[:,1])+1)]
        _pc = PatchCollection(_shapes, 
                              facecolor="w", 
                              alpha=1,
                    edgecolor="w")
        ax.add_collection(_pc)

    if type(shape)==dict:
        if type(Xsize)!=type(None):
            shapes=Parallel(n_jobs=n_jobs)(delayed(_create_polygon)(
                shape[val], 
                x,
                y, 
                wh) for (x, y), wh, val in zip(row_col, _Xsize,Xflatten))

        else:
            shapes=Parallel(n_jobs=n_jobs)(delayed(_create_polygon)(
                shape[val], x,y, 1) for (x, y), val in zip(row_col,Xflatten))
    else:
        if type(Xsize)!=type(None):            
            shapes = [_create_polygon(shape, x,y, wh)
                        for (x, y), wh in zip(row_col, _Xsize)]
            shapes=Parallel(n_jobs=n_jobs)(delayed(_create_polygon)(
                shape, 
                x,
                y, 
                wh) for (x, y), wh in zip(row_col, _Xsize))
        else:
            shapes=Parallel(n_jobs=n_jobs)(delayed(_create_polygon)(
                shape, 
                x,
                y, 
                1) for x, y in row_col)
    # Create patch collection with specified colour/alpha
    if edgecolor==None:
        pc = PatchCollection(shapes, facecolor=facecolors, alpha=1,
                        edgecolor=facecolors)
    else:
        pc = PatchCollection(shapes, facecolor=facecolors, alpha=1,
                        edgecolor=edgecolor)
    pc.set_rasterized(rasterized)
    ax.add_collection(pc)
    
    if row_ticklabels==True:
        ax.yaxis.tick_right()
        _=ax.set_yticks(np.arange(Xshape[0]), labels=rowlabels, rotation=rowrot)
    else:
        ax.set_yticks([])
    if col_ticklabels==True:
        _=ax.set_xticks(np.arange(Xshape[1]), labels=collabels, rotation=colrot)
    else:
        ax.set_xticks([])

def _create_polygon(shape, x, y, r, ry=None, **kwargs):
    if ry==None:
        ry=r
    if shape=="rectangle":
        return Rectangle((x -r/2, y - r/2), r, ry, **kwargs)
    elif shape=="circle":
        return Ellipse((x, y),r, ry, **kwargs)
        
    elif shape=="triangle":
        vnum=3
        rs=[]
        for i in range(vnum):
            rs.append([x+0.5*r*np.cos(np.pi/2+i*2*np.pi/vnum),
                       y+0.5*ry*np.sin(np.pi/2+i*2*np.pi/vnum)])
        return Polygon(rs, **kwargs)
    elif shape=="star":
        rs=[]
        for i in range(5):
            rs.append([x+0.5*r*np.cos(np.pi/2+(2*i)*2*np.pi/5),
                       y+0.5*ry*np.sin(np.pi/2+(2*i)*2*np.pi/5)])
        return Polygon(rs, **kwargs)
    elif shape.startswith("polygon:"):
        _, vnum=shape.split(":")
        vnum=int(vnum)
        rs=[]
        for i in range(vnum):
            rs.append([x+0.5*r*np.cos(np.pi/2+i*2*np.pi/vnum),
                       y+0.5*ry*np.sin(np.pi/2+i*2*np.pi/vnum)])

        return Polygon(rs, **kwargs)
    elif shape.startswith("star:"):
        _, n, m=shape.split(":")
        n, m=int(n), int(m)
        rs=[]
        if n%m==0:
            l=n//m
            for _m in range(m):
                
                for _l in range(l+1):
                    rs.append([x+0.5*r*np.cos(np.pi/2+(_l+_m/m)*2*np.pi/l),
                               y+0.5*ry*np.sin(np.pi/2+(_l+_m/m)*2*np.pi/l)])
        else:
            for i in range(n):
                rs.append([x+0.5*r*np.cos(np.pi/2+(m*i)*2*np.pi/n),
                           y+0.5*ry*np.sin(np.pi/2+(m*i)*2*np.pi/n)])
        return Polygon(rs, **kwargs)
    else:
        raise ValueError("Unknown shape! you gave {}, but it only accepts 'rectangle', 'circle', 'star', 'polygon:n', 'star:n:m'".format(shape))
def _row_plot(df, 
              leaves, 
              fig, 
              category, 
              lut, 
              row_colors, 
              row_plot, 
              row_scatter, 
              row_bar, 
              rowplot_format,
                legend_elements_dict, 
                Xshape, 
                lcatx,
                lcatw,
                yori,
                hmaph, 
                margin, 
                plotkw, 
                scatterkw, 
                barkw ,
                catnum,
                axis_dict, row_axis=0):
    
    def set_axis(_ax, _val, _cat, _margin, _rowplot_format):
        _ax.tick_params(pad=-5)
        _ax.margins(x=_margin,y=_margin)
        _ax.set_xlabel(_cat, rotation=90)
        vmin, vmax=np.amin(_val), np.amax(_val)
        shift=(vmax-vmin)/5
        ax.set_xticks([vmin+shift, vmax-shift],
                        labels=[_rowplot_format.format(x=vmin), 
                        _rowplot_format.format(x=vmax)], 
                        rotation=90)
        ax.set_yticks([])
        
        ax.invert_xaxis()

    row_plot_index=0
    _df=df.iloc[leaves]
    if len(category)!=0:
        for cat in category:
            ax=fig.add_axes([lcatx+row_plot_index*lcatw,yori,lcatw,hmaph])
            facecolors=[lut[cat]["colorlut"][_cat] for _cat in _df[cat]]
            _shapes = [Rectangle(( - 0.5, _y - 0.5), 1.0, 1.0)
                        for _y in range(Xshape[0])]
            _pc = PatchCollection(_shapes, facecolor=facecolors, alpha=1,
                        edgecolor=facecolors)
            ax.add_collection(_pc)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.margins(x=margin,y=margin)
            # ax.axis('off')

            ax.set_xlabel(cat, rotation=90)
            legend_elements=[]
            for _cat, _color in lut[cat]["colorlut"].items():
                legend_elements.append(Line2D([0], [0], marker="s", linewidth=0, markeredgecolor="darkgray",
                                    label=_cat,
                                    markerfacecolor=_color))
            legend_elements_dict[cat]=legend_elements
            row_plot_index+=1
            catnum+=1
            axis_dict["row_"+cat]=ax

    if len(row_colors)!=0:
        for i, (cat, val) in enumerate(row_colors.items()):
            # print(leaves)
            val=np.array(val)[leaves]
            ax=fig.add_axes([lcatx+row_plot_index*lcatw,yori,lcatw,hmaph])
            uniq_labels=np.unique(val)
            _cmap=plt.get_cmap(colormap_list[catnum], len(uniq_labels))
        
            color_lut={u: _cmap(i) for i, u in enumerate(uniq_labels)}
            facecolors=[color_lut[_cat] for _cat in val]
            _shapes = [Rectangle(( - 0.5, _y - 0.5), 1.0, 1.0)
                        for _y in range(Xshape[0])]
            _pc = PatchCollection(_shapes, facecolor=facecolors, alpha=1,
                        edgecolor=facecolors)
            ax.add_collection(_pc)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.margins(x=margin,y=margin)
            # ax.axis('off')

            ax.set_xlabel(cat, rotation=90)
            legend_elements=[]
            for _cat, _color in color_lut.items():
                legend_elements.append(Line2D([0], [0], marker="s", linewidth=0, markeredgecolor="darkgray",
                                    label=_cat,
                                    markerfacecolor=_color))
            legend_elements_dict[cat]=legend_elements
            row_plot_index+=1
            catnum+=1
            axis_dict["row_"+cat]=ax
    if len(row_plot)!=0:
        if type(row_plot)==list:
            for cat in row_plot:
                val=_df[cat]
                ax=fig.add_axes([lcatx+row_plot_index*lcatw,yori,lcatw,hmaph])
                ax.plot(val,np.arange(len(val)),**plotkw)
                set_axis(ax, val, cat, margin, rowplot_format)
                ax.set_ylim(-0.5, val.shape[0]-0.5)
                row_plot_index+=1
                axis_dict["row_"+cat]=ax
        elif type(row_plot)==dict:
            
            for cat, val in row_plot.items():
                val=np.array(val)[leaves]
                ax=fig.add_axes([lcatx+row_plot_index*lcatw,yori,lcatw,hmaph])
                ax.plot(val,np.arange(len(val)),**plotkw)
                ax.set_ylim(-0.5,val.shape[0]-0.5)
                set_axis(ax, val, cat, margin, rowplot_format)
                row_plot_index+=1
                axis_dict["row_"+cat]=ax
    if len(row_scatter)!=0:
        if type(row_scatter)==list:
            for cat in row_scatter:
                val=_df[cat]
                ax=fig.add_axes([lcatx+row_plot_index*lcatw,yori,lcatw,hmaph])
                ax.scatter(val,np.arange(len(val)),**scatterkw)
                ax.set_ylim(-0.5, val.shape[0]-0.5)
                set_axis(ax, val, cat, margin, rowplot_format)
                row_plot_index+=1
                axis_dict["row_"+cat]=ax
        elif type(row_scatter)==dict:
            
            for cat, val in row_scatter.items():
                val=np.array(val)[leaves]
                ax=fig.add_axes([lcatx+row_plot_index*lcatw,yori,lcatw,hmaph])
                ax.scatter(val,np.arange(len(val)),**scatterkw)
                ax.set_ylim(-0.5, val.shape[0]-0.5)
                set_axis(ax, val, cat, margin, rowplot_format)
                row_plot_index+=1
                axis_dict["row_"+cat]=ax
    if len(row_bar)!=0:
        if type(row_bar)==list:
            for cat in row_bar:
                val=_df[cat]
                ax=fig.add_axes([lcatx+row_plot_index*lcatw,yori,lcatw,hmaph])
                ax.barh(np.arange(len(val)), width=val,**barkw)
                set_axis(ax, val, cat, margin, rowplot_format)
                row_plot_index+=1
                axis_dict["row_"+cat]=ax
        elif type(row_bar)==dict:
            
            for cat, val in row_bar.items():
                val=np.array(val)[leaves]
                ax=fig.add_axes([lcatx+row_plot_index*lcatw,yori,lcatw,hmaph])
                ax.barh(np.arange(len(val)), width=val,**barkw)
                set_axis(ax, val, cat, margin, rowplot_format)
                row_plot_index+=1
                axis_dict["row_"+cat]=ax

    if row_axis!=0:
        for i in range(row_axis):
            ax=fig.add_axes([lcatx+row_plot_index*lcatw,yori,lcatw,hmaph])
            
            set_axis(ax, val, cat, margin, rowplot_format)
            row_plot_index+=1
            axis_dict["row_"+str(i)]=ax


    return legend_elements_dict, catnum, axis_dict



def _col_plot(leaves, fig, col_colors, col_plot, 
              
              col_scatter, col_bar, 
              colplot_format,
                legend_elements_dict, 
                Xshape, tcaty,tcatw,hmapx,
                hmapw, margin, plotkw, scatterkw, barkw ,catnum, axis_dict,col_axis=0):
    
    def set_axis(_ax, _val, _cat, _margin, _lowplot_format, pad=-5):
        _ax.tick_params(pad=pad)
        _ax.margins(x=_margin,y=_margin)
        _ax.set_ylabel(_cat, rotation=0,labelpad=5, ha="right")
        vmin, vmax=np.amin(_val), np.amax(_val)
        shift=(vmax-vmin)/5

        _ax.set_yticks([vmin+shift, vmax-shift],
                        labels=[_lowplot_format.format(x=vmin), 
                        _lowplot_format.format(x=vmax)], 
                        rotation=0)
        _ax.set_xticks([])


    row_plot_index=0
    if len(col_colors)!=0:
        for i, (cat, val) in enumerate(col_colors.items()):
            # print(leaves)
            val=np.array(val)[leaves]
            ax=fig.add_axes([hmapx,tcaty+row_plot_index*tcatw,hmapw,tcatw])
            # ax.tick_params(pad=15)
            uniq_labels=np.unique(val)
            _cmap=plt.get_cmap(colormap_list[catnum], len(uniq_labels))
        
            color_lut={u: _cmap(i) for i, u in enumerate(uniq_labels)}
            facecolors=[color_lut[_cat] for _cat in val]
            _shapes = [Rectangle(( _y - 0.5,  - 0.5), 1.0, 1.0)
                        for _y in range(Xshape[1])]
            _pc = PatchCollection(_shapes, facecolor=facecolors, alpha=1,
                        edgecolor=facecolors)
            ax.add_collection(_pc)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.margins(x=margin,y=margin)
            # ax.axis('off')

            ax.set_ylabel(cat, rotation=0,labelpad=0, ha="right")
            legend_elements=[]
            for _cat, _color in color_lut.items():
                legend_elements.append(Line2D([0], [0], marker="s", linewidth=0, markeredgecolor="darkgray",
                                    label=_cat,
                                    markerfacecolor=_color))
            legend_elements_dict[cat]=legend_elements
            row_plot_index+=1
            catnum+=1
            axis_dict["col_"+cat]=ax
    if len(col_plot)!=0:
        if type(col_plot)==list:
            raise Exception("Currently, 'col_plot' option accepts only dictionary")
        elif type(col_plot)==dict:
            
            for cat, val in col_plot.items():
                val=np.array(val)[leaves]
                ax=fig.add_axes([hmapx,tcaty+row_plot_index*tcatw,hmapw,tcatw])
                ax.plot(val,np.arange(len(val)),**plotkw)
                set_axis(ax, val, cat, margin, colplot_format)
                row_plot_index+=1
                axis_dict["col_"+cat]=ax
    if len(col_scatter)!=0:
        if type(col_scatter)==list:
            raise Exception("Currently, 'col_scatter' option accepts only dictionary")

        elif type(col_scatter)==dict:
            
            for cat, val in col_scatter.items():
                val=np.array(val)[leaves]
                ax=fig.add_axes([hmapx,tcaty+row_plot_index*tcatw,hmapw,tcatw])
                ax.scatter(val,np.arange(len(val)),**scatterkw)
                set_axis(ax, val, cat, margin, colplot_format)
                row_plot_index+=1
                axis_dict["col_"+cat]=ax
    if len(col_bar)!=0:
        if type(col_bar)==list:
            raise Exception("Currently, 'col_scatter' option accepts only dictionary")

        elif type(col_bar)==dict:
            
            for cat, val in col_bar.items():
                val=np.array(val)[leaves]
                ax=fig.add_axes([hmapx,tcaty+row_plot_index*tcatw,hmapw,tcatw])
                ax.bar(np.arange(len(val)), height=val,**barkw)
                set_axis(ax, val, cat, margin, colplot_format)
                row_plot_index+=1
                axis_dict["col_"+cat]=ax

    if col_axis!=0:
        for i in range(col_axis):
            ax=fig.add_axes([hmapx,tcaty+row_plot_index*tcatw,hmapw,tcatw])
            set_axis(ax, val, cat, margin, colplot_format)
            row_plot_index+=1
            axis_dict["col_"+str(i)]=ax



    return legend_elements_dict, catnum, axis_dict


def _process_vdata(df, variables,category ,row,  col,rowlabels, sizes, size_title, colors, lut,fillna, ztranform,cunit):
    Xsize=None
    if len(variables)==0 and len(category)==0 and len(row)==0 and len(col)==0:
        print("Assuming data is a wide form")
        if rowlabels=="":
            rowlabels=np.array(df.index)
        else:
            rowlabels=np.array(df[rowlabels])
            df=df.drop([rowlabels], axis=1)
        X=df.to_numpy()
        if type(sizes)==pd.DataFrame:
            Xsize=sizes.to_numpy()
            if size_title=="":
                try:
                    size_title=sizes.name
                except:
                    size_title=""
        elif type(sizes)==dict:
            size_title, Xsize, =list(sizes.keys())[0], list(sizes.values())[0]
        elif type(sizes)==np.ndarray or type(sizes)==list:
            Xsize=np.array(sizes)
        
        collabels=np.array(df.columns)
    elif len(row)==0 and len(col)==0:
        print("Assuming data is a wide form")
        if rowlabels=="":
            rowlabels=np.array(df.index)
        else:
            rowlabels=np.array(df[rowlabels])
            df=df.drop([rowlabels], axis=1)
        X, category=_separate_data(df, variables, category)
        
        for cat, _palette in zip(category, colormap_list):
            _clut, _mlut=_create_color_markerlut(df, cat,_palette,marker_list)
            lut[cat]={"colorlut":_clut, "markerlut":_mlut}
        if type(sizes)==pd.DataFrame:
            Xsize=sizes.to_numpy()
        elif type(sizes)==np.ndarray or type(sizes)==list:
            Xsize=np.array(sizes)
        elif type(sizes)==str and sizes!="":
            Xsize=df[sizes].values

        if len(variables)>0:
            collabels=np.array(variables)
        else:
            collabels=np.array(df.drop(category, axis=1).columns)
    elif len(variables)==0 and len(category)==0:
        print("Assuming data is a long form")
        _df=df.pivot_table(index=row, columns=col, values=colors)
        _df=_df.fillna(fillna)
        X=_df.to_numpy()
        if type(sizes)==str and sizes !="":
            _df=df.pivot_table(index=row, columns=col, values=sizes)
            Xsize=_df.to_numpy()

        if type(rowlabels)==str and rowlabels=="":
            rowlabels=np.array(_df.index)
        else:
            rowlabels=np.array(rowlabels)
        collabels=np.array(df.columns)
    else:
        raise Exception("")
    rowlabels=rowlabels.astype(str)
    if ztranform==True:
        X=zscore(X, axis=0)
        if cunit =="":
            cunit="zscore"
    return df, X, category, rowlabels, collabels, Xsize, size_title, cunit, lut


def _process_cdata(df, variables,category ,row,  col,rowlabels, sizes, size_title, colors, lut,fillna,cunit):
    Xsize=None
    if len(variables)==0 and len(category)==0 and len(row)==0 and len(col)==0:
        print("Assuming data is a wide form")
        if rowlabels=="":
            rowlabels=np.array(df.index)
        else:
            rowlabels=np.array(df[rowlabels])
            df=df.drop([rowlabels], axis=1)
        X=df.to_numpy()
        if type(sizes)==pd.DataFrame:
            Xsize=sizes.to_numpy()
            if size_title=="":
                try:
                    size_title=sizes.name
                except:
                    size_title=""
        elif type(sizes)==dict:
            size_title, Xsize, =list(sizes.keys())[0], list(sizes.values())[0]
        elif type(sizes)==np.ndarray or type(sizes)==list:
            Xsize=np.array(sizes)
        
        collabels=np.array(df.columns)
    elif len(row)==0 and len(col)==0:
        print("Assuming data is a wide form")
        if rowlabels=="":
            rowlabels=np.array(df.index)
        else:
            rowlabels=np.array(df[rowlabels])
            df=df.drop([rowlabels], axis=1)
    
        X, category=_separate_cdata(df, variables, category)
        
        for cat, _palette in zip(category, colormap_list):
            _clut, _mlut=_create_color_markerlut(df, cat,_palette,marker_list)
            lut[cat]={"colorlut":_clut, "markerlut":_mlut}
        if type(sizes)==pd.DataFrame:
            Xsize=sizes.to_numpy()
        elif type(sizes)==np.ndarray or type(sizes)==list:
            Xsize=np.array(sizes)
        elif type(sizes)==str and sizes!="":
            Xsize=df[sizes].values

        if len(variables)>0:
            collabels=np.array(variables)
        else:
            collabels=np.array(df.drop(category, axis=1).columns)
    elif len(variables)==0 and len(category)==0:
        print("Assuming data is a long form")
        _df=df.pivot_table(index=row, columns=col, values=colors)
        _df=_df.fillna(fillna)
        X=_df.to_numpy()
        if type(sizes)==str and sizes !="":
            _df=df.pivot_table(index=row, columns=col, values=sizes)
            Xsize=_df.to_numpy()

        if type(rowlabels)==str and rowlabels=="":
            rowlabels=np.array(_df.index)
        else:
            rowlabels=np.array(rowlabels)
        collabels=np.array(df.columns)
    else:
        raise Exception("")
    rowlabels=rowlabels.astype(str)

    return df, X, category, rowlabels, collabels, Xsize, size_title, cunit, lut


def _dendrogram(X: np.ndarray, Xsize: np.ndarray, ax: plt.Axes, rowlabels,treepalette ,approx_clusternum, metric,method,above_threshold_color,orientation):

    _tcmap=plt.get_cmap(treepalette)
    tcmap = _tcmap(np.linspace(0, 1, approx_clusternum))
    sch.set_link_color_palette([mpl.colors.rgb2hex(rgb[:3]) for rgb in tcmap])
    if orientation=="left":
        D=squareform(pdist(X,metric=metric))
    elif orientation=="top":
        D=squareform(pdist(X.T,metric=metric))
    Y = fcl.linkage(D, method=method)
    Z = sch.dendrogram(Y,orientation=orientation,above_threshold_color=above_threshold_color, no_plot=True)
    t=_dendrogram_threshold(Z,approx_clusternum)

    Z = sch.dendrogram(Y,ax=ax,orientation=orientation,labels=rowlabels,color_threshold=t,above_threshold_color=above_threshold_color)
    # Sorting the row values of X.
    if orientation=="left":
        X=X[Z['leaves']]
        if type(Xsize)!=type(None):
            Xsize=Xsize[Z['leaves']]
    elif orientation=="top":
        X=X[:, Z['leaves']]
        if type(Xsize)!=type(None):
            Xsize=Xsize[:, Z['leaves']]
    rclusters = _get_cluster_classes(Z, original_rows=[])
    # print(_rlabels)
    rowlabels=rowlabels[Z['leaves']]
    return X, Xsize, Z, rowlabels, rclusters



class _AnyObject:
    pass


class _AnyObjectHandler:
    def __init__(self, facecolor, shape):
        self.facecolor=facecolor
        self.shape=shape
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        # print(x0, y0)
        width, height = handlebox.width, handlebox.height
        # print(width, height)
        patch=_create_polygon(self.shape, 10, 3, 10, facecolor=self.facecolor,
                                   edgecolor='black', lw=1,
                                   transform=handlebox.get_transform())
        # patch = Ellipse([10, 3], 10, 10, facecolor=self.facecolor,
        #                            edgecolor='black', lw=1,
        #                            transform=handlebox.get_transform())
        handlebox.add_artist(patch)
        
        return patch
    


def _axis_loc_sizes(col_ticklabels: bool, 
                    row_ticklabels: bool,
                    boxlabels: bool, 
                    row_cluster: bool,
                    col_cluster: bool,
                    show_legend: int,
                    rowplot_num: int,
                    colplot_num: int,
                    collabels: np.ndarray,
                    Xshape: Union[np.ndarray, list, tuple]) -> List:
    # determining the size and positionn of axes


    if col_ticklabels==True:
        lmax = float(np.amax(list(map(len, list(collabels.astype(str)))))) 
        lmax=np.amin([lmax/150, 0.3])
    else:
        lmax=0

    if boxlabels==True:
        row_ticklabels=False

    xori=0.05
    yori=0.11+lmax
    lcatw=0.04
    
    legendw=0.15*show_legend
    if boxlabels==True:
        boxwidth=0.15
    else:
        boxwidth=0.
    if row_cluster==True:
        ltreew=0.15
        ttreew=0.65-lcatw*rowplot_num-legendw-boxwidth
    else:
        ltreew=0
        ttreew=0.75-lcatw*rowplot_num-legendw
    
    if col_cluster==True:
        ltreeh=0.75-lmax
        ttreeh=0.05
    else:
        ltreeh=0.8-lmax
        ttreeh=0
    if ttreew<0:
        raise Exception("Too many things to plot. \
                        Please reduce the number of \
                        row-wise plots.")
    hmapw=ttreew
    hmaph=ltreeh
    lcatx=xori+ltreew
    tcaty=yori+ltreeh
    tcath=0.025

    hmapx=xori+ltreew+lcatw*rowplot_num
    ttreey=yori+ltreeh+tcath*colplot_num
    legendh=(3/Xshape[0])*hmaph

    return (xori, yori, 
            boxwidth, 
            legendw, legendh,
            ltreew, ltreeh,  
            ttreey, ttreew,  ttreeh, 
            hmapx, hmapw, hmaph, 
            lcatx, lcatw, 
            tcaty, tcath, 
            row_ticklabels)


def _create_shape_legend_elements(Xsize, 
                                  Xshape,
                                  hmapw,
                                  hmaph,
                                  legendw,
                                  legendh,
                                  _scale_size,
                                  _reverse_size,
                                  size_legend_num,
                                  shape,
                                  size_format):
    size_legend_elements=[]
    smin=np.amin(Xsize)
    smax=np.amax(Xsize)
    _scaled=_scale_size(Xsize,1, smin, smax)
    vmin, vmax=np.amin(_scaled), np.amax(_scaled)
    # print("scaled: ", vmin, vmax)
    vinterval=(vmax-vmin)/(size_legend_num-1)
    
    if size_format=="":
        if 1<np.abs(smax)<=1000:
            size_format="{x:.2f}"
        elif 0<np.abs(smax)<=1 or 1000<np.abs(smax):
            size_format="{x:.3E}"
    
    

    sx=1
    size_legend_elements.append(Rectangle((0 -0.5,0-0.5), 1, size_legend_num))

    size_labels=[]
    prev_top=0
    for _i in range(size_legend_num):
        if _i==size_legend_num-1:
            s=vmax
        else:
            s=vmin+_i*vinterval
        if s <0.1:
            s=0.1
        sx=s*(hmapw/legendw)/Xshape[1]
        sy=s*(hmaph/legendh)*size_legend_num/Xshape[0]
        
        if shape=="by_category":
            size_legend_elements.append(_create_polygon("circle", 0, _i, sx,ry=sy))
        else:
            size_legend_elements.append(_create_polygon(shape, 0, _i, sx,ry=sy))


        prev_top+=sy+0.1

        _s=_reverse_size(s, 1, smin, smax)
        size_labels.append([size_format.format(x=_s), _i])
    

    return size_legend_elements, size_labels