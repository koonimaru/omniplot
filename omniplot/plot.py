"""
           /\   /\
          / \  / \
       @@@@@@@@@@@@@@
      @@@@@@@@@@@@@@@@
       (  ---  ---   )
       (   O L  O    )3
        (   <--->   )
         `---------'
        /             \  
       /  /|        |\ \
           |        |
"""
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
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.manifold import (
    Isomap,
    LocallyLinearEmbedding,
    MDS,
    SpectralEmbedding,
    TSNE,
)
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.random_projection import SparseRandomProjection
import sys 
import matplotlib as mpl
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform

from sklearn.decomposition import PCA, NMF
import umap 
from scipy.stats import zscore
from itertools import combinations

colormap_list=["nipy_spectral", "terrain","tab20b","gist_rainbow","CMRmap","coolwarm","gnuplot","gist_stern","brg","rainbow"]
plt.rcParams['font.family']= 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['svg.fonttype'] = 'none'
sns.set_theme()

def _calc_curveture(normx, normy):
    perp=[]
    for i, (nx, ny) in enumerate(zip(normx, normy)):
        if i==0:
            perp.append(0)
            continue
        r=(nx**2+ny**2)**0.5
        sina=ny/r
        cosa=nx/r
        sinamb=sina*np.cos(np.pi*0.25)-cosa*np.sin(np.pi*0.25)
        perp.append(r*sinamb)
    perp=np.array(perp)
    return perp

def _get_cluster_classes(den, label='ivl'):
    cluster_idxs = defaultdict(list)
    for c, pi in zip(den['color_list'], den['icoord']):
        for leg in pi[1:3]:
            i = (leg - 5.0) / 10.0
            if abs(i - int(i)) < 1e-5:
                cluster_idxs[c].append(int(i))

    cluster_classes = {}
    for c, l in cluster_idxs.items():
        i_l = [den[label][i] for i in l]
        cluster_classes[c] = i_l

    return cluster_classes

def _dendrogram_threshold(Z, approx_clusternum):
    lbranches=np.array(Z["dcoord"])[:,:2]
    rbranches=np.array(Z["dcoord"])[:,2:]
    thre=np.linspace(0, np.amax(Z["dcoord"]), 100)[::-1]
    for t in thre:
        crossbranches=np.sum(lbranches[:,1]>t)+np.sum(rbranches[:,0]>t)-np.sum(lbranches[:,0]>t)-np.sum(rbranches[:,1]>t)
        if crossbranches>=approx_clusternum:
            break 
    return t

def dotplot(df: pd.DataFrame,
            row: str="",
            col: str="",
            dfc=pd.DataFrame(),
            scaling: float=10,
            color_val: str="",
            size_val: str="",
            highlight: str="",
            color_title: str="",
            size_title: str="",
            save: str="",
            threshold: float=-np.log10(0.05),
            row_clustering: bool=True,
            xtickrotation: float=90,
            column_order: list=[],
            show: bool=False
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
    axes: dict
    
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
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
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
        viridis = cm.get_cmap('viridis', 12)
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
    fig = plt.figure()
    fig.set_figheight(6)
    fig.set_figwidth(6)
    ax1 = plt.subplot2grid(shape=(6, 6), loc=(0, 0), colspan=4, rowspan=6)
    ax2 = plt.subplot2grid(shape=(6, 6), loc=(0, 4), colspan=2, rowspan=2)
    ax3 = plt.subplot2grid(shape=(6, 6), loc=(2, 4), colspan=2, rowspan=1)
 
    collection = mc.CircleCollection(sizes,
                                     edgecolors=edge_colors, 
                                     offsets=xy, 
                                     transOffset=ax1.transData, 
                                     facecolors=_colors,
                                     linewidths=2)
    ax1.add_collection(collection)
    ax1.margins(0.1)
    ax1.set_xlim(-1,len(_x)+1)
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
    
    lxy=[[0.25, i*1] for i in range(3)]
    collection2 = mc.CircleCollection([middle0*scaling,middle1*scaling, maxsize*scaling], offsets=lxy, transOffset=ax2.transData, facecolors='gray',edgecolors="black")
    ax2.add_collection(collection2)
    ax2.axis('off')
    ax2.margins(0.3)
    for text, (x, y) in zip([middle0,middle1, maxsize], lxy):
        ax2.text(x+0.25, y,str(text),va="center")
    if size_title=="":
        size_title=size_val
    ax2.text(0.5,-0.5, size_title,va="center",ha="center")
    #ax[1].set_yticks(np.arange(3))
    #ax[1].set_yticklabels([minsize,middle, maxsize], rotation=0)
    #plt.tight_layout()
    if save!="":
        plt.savefig(save+".svg")
    if show==True:
        plt.show()
    else:
        return {"ax1":ax1,"ax2":ax2,"ax3":ax3}


def radialtree(df,n_clusters: int=3,
               category: List[str]=[], **kwargs):
    
    if len(category) !=0:
        category_df=df[category]
        df=df.drop(category, axis=1)
        X = df.values
        print(X)
        assert X.dtype==float, f"data must contain only float values except {category} column."
        
    else:    
        X = df.values
        assert X.dtype==float, "data must contain only float values."
    X=zscore(X, axis=0)
    D=squareform(pdist(X))
    Y = sch.linkage(D, method='ward')
    Z = sch.dendrogram(Y,labels=df.index,no_plot=True)
    t=_dendrogram_threshold(Z, n_clusters)
    Z=sch.dendrogram(Y,
                        labels = df.index,
                        color_threshold=t,no_plot=True)
    sample_classes={k: list(category_df[k]) for k in category_df.columns}
    _radialtree(Z, sample_classes=sample_classes,addlabels=False, **kwargs)
    

def _radialtree(Z2,fontsize: int=8,
               figsize: Optional[list]=None,
               palette: str="gist_rainbow", 
               addlabels: bool=True, 
               show: bool=False,
               sample_classes: Optional[dict]=None,
               colorlabels: Optional[dict]=None,
         colorlabels_legend: Optional[dict]=None
         ) -> plt.Axes:
    """
    Drawing a radial dendrogram from a scipy dendrogram output.
    Parameters
    ----------
    Z2 : dictionary
        A dictionary returned by scipy.cluster.hierarchy.dendrogram
    addlabels: bool
        A bool to choose if labels are shown.
    fontsize : float
        A float to specify the font size
    figsize : [x, y] array-like
        1D array-like of floats to specify the figure size
    palette : string
        Matlab colormap name.
    sample_classes : dict
        A dictionary that contains lists of sample subtypes or classes. These classes appear 
        as color labels of each leaf. Colormaps are automatically assigned. Not compatible 
        with options "colorlabels" and "colorlabels_legend".
        e.g., {"color1":["Class1","Class2","Class1","Class3", ....]} 
    colorlabels : dict
        A dictionary to set color labels to leaves. The key is the name of the color label. 
        The value is the list of RGB color codes, each corresponds to the color of a leaf. 
        e.g., {"color1":[[1,0,0,1], ....]}   
    colorlabels_legend : dict
        A nested dictionary to generate the legends of color labels. The key is the name of 
        the color label. The value is a dictionary that has two keys "colors" and "labels". 
        The value of "colors" is the list of RGB color codes, each corresponds to the class of a leaf. 
        e.g., {"color1":{"colors":[[1,0,0,1], ....], "labels":["label1","label2",...]}}   
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
    if figsize==None and colorlabels != None:
        figsize=[7,5]
    elif figsize==None and sample_classes != None:
        figsize=[7,5]
    elif figsize==None :
        figsize=[5,5]
    linewidth=0.5
    R=1
    width=R*0.1
    space=R*0.05
    if colorlabels != None:
        offset=width*len(colorlabels)/R+space*(len(colorlabels)-1)/R+0.05
        print(offset)
    elif sample_classes != None:
        offset=width*len(sample_classes)/R+space*(len(sample_classes)-1)/R+0.05
        print(offset)
    else:
        offset=0
    
    xmax=np.amax(Z2['icoord'])
    ymax=np.amax(Z2['dcoord'])
    
    ucolors=sorted(set(Z2["color_list"]))
    #cmap = cm.gist_rainbow(np.linspace(0, 1, len(ucolors)))
    cmp=cm.get_cmap(palette, len(ucolors))
    #print(cmp)
    if type(cmp) == matplotlib.colors.LinearSegmentedColormap:
        cmap = cmp(np.linspace(0, 1, len(ucolors)))
    else:
        cmap=cmp.colors
    fig, ax=plt.subplots(figsize=figsize)
    i=0
    label_coords=[]
    for x, y, c in sorted(zip(Z2['icoord'], Z2['dcoord'],Z2["color_list"])):
    #x, y = Z2['icoord'][0], Z2['dcoord'][0]
        _color=cmap[ucolors.index(c)]
        if c=="C0": #np.abs(_xr1)<0.000000001 and np.abs(_yr1) <0.000000001:
            _color="black"
        
        # transforming original x coordinates into relative circumference positions and y into radius
        # the rightmost leaf is going to [1, 0] 
        r=R*(1-np.array(y)/ymax)
        _x=np.cos(2*np.pi*np.array([x[0],x[2]])/xmax) # transforming original x coordinates into x circumference positions
        _xr0=_x[0]*r[0]
        _xr1=_x[0]*r[1]
        _xr2=_x[1]*r[2]
        _xr3=_x[1]*r[3]
        _y=np.sin(2*np.pi*np.array([x[0],x[2]])/xmax) # transforming original x coordinates into y circumference positions
        _yr0=_y[0]*r[0]
        _yr1=_y[0]*r[1]
        _yr2=_y[1]*r[2]
        _yr3=_y[1]*r[3]
        #plt.scatter([_xr0, _xr1, _xr2, _xr3],[_yr0, _yr1, _yr2,_yr3], c="b")
        
        
        #if y[0]>0 and y[3]>0:
            #_color="black"
        #plotting radial lines
        plt.plot([_xr0, _xr1], [_yr0, _yr1], c=_color,linewidth=linewidth, rasterized=True)
        plt.plot([_xr2, _xr3], [_yr2,_yr3], c=_color,linewidth=linewidth, rasterized=True)
        
        #plotting circular links between nodes
        if _yr1> 0 and _yr2>0:
            link=np.sqrt(r[1]**2-np.linspace(_xr1, _xr2, 100)**2)
            plt.plot(np.linspace(_xr1, _xr2, 100), link, c=_color,linewidth=linewidth, rasterized=True)
        elif _yr1 <0 and _yr2 <0:
            link=-np.sqrt(r[1]**2-np.linspace(_xr1, _xr2, 100)**2)
            
            plt.plot(np.linspace(_xr1, _xr2, 100), link, c=_color,linewidth=linewidth, rasterized=True)
        elif _yr1> 0 and _yr2 < 0:
            _r=r[1]
            if _xr1 <0 or _xr2 <0:
                _r=-_r
            link=np.sqrt(r[1]**2-np.linspace(_xr1, _r, 100)**2)
            plt.plot(np.linspace(_xr1, _r, 100), link, c=_color,linewidth=linewidth, rasterized=True)
            link=-np.sqrt(r[1]**2-np.linspace(_r, _xr2, 100)**2)
            plt.plot(np.linspace(_r, _xr2, 100), link, c=_color,linewidth=linewidth, rasterized=True)
        
        #Calculating the x, y coordinates and rotation angles of labels
        
        if y[0]==0:
            label_coords.append([(1.05+offset)*_xr0, (1.05+offset)*_yr0,360*x[0]/xmax])
            #plt.text(1.05*_xr0, 1.05*_yr0, Z2['ivl'][i],{'va': 'center'},rotation_mode='anchor', rotation=360*x[0]/xmax)
            i+=1
        if y[3]==0:
            label_coords.append([(1.05+offset)*_xr3, (1.05+offset)*_yr3,360*x[2]/xmax])
            #plt.text(1.05*_xr3, 1.05*_yr3, Z2['ivl'][i],{'va': 'center'},rotation_mode='anchor', rotation=360*x[2]/xmax)
            i+=1
    

    if addlabels==True:
        assert len(Z2['ivl'])==len(label_coords), "Internal error, label numbers "+str(len(Z2['ivl'])) +" and "+str(len(label_coords))+" must be equal!" 
        
        #Adding labels
        for (_x, _y,_rot), label in zip(label_coords, Z2['ivl']):
            plt.text(_x, _y, label,{'va': 'center'},rotation_mode='anchor', rotation=_rot,fontsize=fontsize)
    
    
    
    if colorlabels != None:
        assert len(Z2['ivl'])==len(label_coords), "Internal error, label numbers "+str(len(Z2['ivl'])) +" and "+str(len(label_coords))+" must be equal!" 
        
        j=0
        outerrad=R*1.05+width*len(colorlabels)+space*(len(colorlabels)-1)
        print(outerrad)
        #sort_index=np.argsort(Z2['icoord'])
        #print(sort_index)
        intervals=[]
        for i in range(len(label_coords)):
            _xl,_yl,_rotl =label_coords[i-1]
            _x,_y,_rot =label_coords[i]
            if i==len(label_coords)-1:
                _xr,_yr,_rotr =label_coords[0]
            else:
                _xr,_yr,_rotr =label_coords[i+1]
            d=((_xr-_xl)**2+(_yr-_yl)**2)**0.5
            intervals.append(d)
        colorpos=intervals#np.ones([len(label_coords)])
        labelnames=[]
        for labelname, colorlist in colorlabels.items():
            colorlist=np.array(colorlist)[Z2['leaves']]
            if j!=0:
                outerrad=outerrad-width-space
            innerrad=outerrad-width
            patches, texts =plt.pie(colorpos, colors=colorlist,
                    radius=outerrad,
                    counterclock=True,
                    startangle=label_coords[0][2]*0.5)
            circle=plt.Circle((0,0),innerrad, fc='white')
            plt.gca().add_patch(circle)
            labelnames.append(labelname)
            j+=1
        
        if colorlabels_legend!=None:
            for i, labelname in enumerate(labelnames):
                print(colorlabels_legend[labelname]["colors"])
                colorlines=[]
                for c in colorlabels_legend[labelname]["colors"]:
                    colorlines.append(Line2D([0], [0], color=c, lw=4))
                leg=plt.legend(colorlines,
                           colorlabels_legend[labelname]["labels"],
                       bbox_to_anchor=(1.5+0.3*i, 1.0),
                       title=labelname)
                plt.gca().add_artist(leg)   
    elif sample_classes!=None:
        assert len(Z2['ivl'])==len(label_coords), "Internal error, label numbers "+str(len(Z2['ivl'])) +" and "+str(len(label_coords))+" must be equal!" 
        
        j=0
        outerrad=R*1.05+width*len(sample_classes)+space*(len(sample_classes)-1)
        print(outerrad)
        #sort_index=np.argsort(Z2['icoord'])
        #print(sort_index)
        intervals=[]
        for i in range(len(label_coords)):
            _xl,_yl,_rotl =label_coords[i-1]
            _x,_y,_rot =label_coords[i]
            if i==len(label_coords)-1:
                _xr,_yr,_rotr =label_coords[0]
            else:
                _xr,_yr,_rotr =label_coords[i+1]
            d=((_xr-_xl)**2+(_yr-_yl)**2)**0.5
            intervals.append(d)
        colorpos=intervals#np.ones([len(label_coords)])
        labelnames=[]
        colorlabels_legend={}
        for labelname, colorlist in sample_classes.items():
            ucolors=sorted(list(np.unique(colorlist)))
            type_num=len(ucolors)
            _cmp=cm.get_cmap(colormap_list[j])
            _colorlist=[_cmp(ucolors.index(c)/(type_num-1)) for c in colorlist]
            _colorlist=np.array(_colorlist)[Z2['leaves']]
            if j!=0:
                outerrad=outerrad-width-space
            innerrad=outerrad-width
            print(outerrad, innerrad)
            patches, texts =plt.pie(colorpos, colors=_colorlist,
                    radius=outerrad,
                    counterclock=True,
                    startangle=label_coords[0][2]*0.5)
            circle=plt.Circle((0,0),innerrad, fc='white')
            plt.gca().add_patch(circle)
            labelnames.append(labelname)
            colorlabels_legend[labelname]={}
            colorlabels_legend[labelname]["colors"]=_cmp(np.linspace(0, 1, type_num))
            colorlabels_legend[labelname]["labels"]=ucolors
            j+=1
        
        if colorlabels_legend!=None:
            for i, labelname in enumerate(labelnames):
                print(colorlabels_legend[labelname]["colors"])
                colorlines=[]
                for c in colorlabels_legend[labelname]["colors"]:
                    colorlines.append(Line2D([0], [0], color=c, lw=4))
                leg=plt.legend(colorlines,
                           colorlabels_legend[labelname]["labels"],
                       bbox_to_anchor=(1.1, 1.0-0.3*i),
                       title=labelname)
                plt.gca().add_artist(leg)
            #break
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.bottom.set_visible(False)
    ax.set_rasterization_zorder(None)
    plt.xticks([])
    plt.yticks([])
    
    if colorlabels!=None:
        maxr=R*1.05+width*len(colorlabels)+space*(len(colorlabels)-1)
    elif sample_classes !=None:
        maxr=R*1.05+width*len(sample_classes)+space*(len(sample_classes)-1)
    else:
        maxr=R*1.05
    plt.xlim(-maxr,maxr)
    plt.ylim(-maxr,maxr)
    plt.subplots_adjust(left=0.05, right=0.85)
    if show==True:
        plt.show()
    else:
        return ax
    


def _complex_clustermap(df,row_plot=[],col_plot=[],approx_clusternum=10,color_var=0,merginalsum=False,show=True,method="ward", **kwargs):
    rnum, cnum=df.shape
    sns.set(font_scale=1)
    
    if merginalsum:
        white_bgr=np.ones([rnum, 4])
        white_bgc=np.ones([cnum, 4])
        g=sns.clustermap(df,col_colors=white_bgc, row_colors=white_bgr,method=method,**kwargs)
        mat=df.to_numpy()
        r=np.sum(mat, axis=1)
        g.ax_row_colors.barh(np.arange(r.shape[0])+0.5, r[leaves_list(g.dendrogram_row.linkage)]/np.amax(r))
        g.ax_row_colors.invert_xaxis()
        
        c=np.sum(mat, axis=0)
        #print(leaves_list(g.dendrogram_col.linkage))
        g.ax_col_colors.bar(np.arange(c.shape[0])+0.5,c[leaves_list(g.dendrogram_col.linkage)]/np.amax(c))
        g.ax_col_colors.invert_yaxis()
    
    elif len(row_plot)>0:
        
        white_bg=[np.ones([rnum, 4]) for _ in range(len(row_plot))]
        g=sns.clustermap(df, row_colors=white_bg,method=method,**kwargs)#hot(cbenign))
        for i, r in enumerate(row_plot):
            g.ax_row_colors.plot(r[leaves_list(g.dendrogram_row.linkage)]/np.amax(r)+i, np.arange(r.shape[0])+0.5)
        g.ax_row_colors.invert_xaxis()
        
    else:
        g=sns.clustermap(df,method=method,**kwargs)
    if color_var>0:
        cmap = cm.nipy_spectral(np.linspace(0, 1, color_var))
    else:
        cmap = cm.nipy_spectral(np.linspace(0, 1, approx_clusternum+5))
    hierarchy.set_link_color_palette([mpl.colors.rgb2hex(rgb[:3]) for rgb in cmap])
    lbranches=np.array(g.dendrogram_row.dendrogram["dcoord"])[:,:2]
    rbranches=np.array(g.dendrogram_row.dendrogram["dcoord"])[:,2:]
    thre=np.linspace(0, np.amax(g.dendrogram_row.dendrogram["dcoord"]), 100)[::-1]
    for t in thre:
        #print(np.sum(lbranches[:,1]>t),np.sum(rbranches[:,0]>t),np.sum(lbranches[:,0]>t),np.sum(rbranches[:,1]>t))
        crossbranches=np.sum(lbranches[:,1]>t)+np.sum(rbranches[:,0]>t)-np.sum(lbranches[:,0]>t)-np.sum(rbranches[:,1]>t)
        #print(crossbranches)
        
        if crossbranches>approx_clusternum:
            break
    
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
    if show:
        plt.show()
    return pd.DataFrame(cdata), g

def complex_clustermap(df,
                       row_colormap={},
                       col_colormap={},
                       row_plot={},
                       col_plot={},
                       row_color_legend={},
                       col_color_legend={},
                       approx_clusternum=10,
                       approx_clusternum_col=3,
                       color_var=0,
                       merginalsum=False,
                       show=False,
                       method="ward",
                       return_col_cluster=True, 
                       **kwargs):
    """
    Drawing a clustered heatmap with merginal plots.
    
    Parameters
    ----------
    df : pandas DataFrame
    row_colormap: dict
        the column name of a category that is going to be placed in the row of the dotplot
    col_colormap: dict
        the column name of a category that is going to be placed in the column of the dotplot
    row_plot : dict
        The column name for the values represented as dot colors.
    col_plot : dict
        The column name for the values represented as dot sizes. 
    row_color_legend: dict
        The scale of dots. If resulting dots are too large (or small), you can reduce (or increase) dot sizes by adjusting this value.
    col_color_legend: dict
        The scale of dots. If resulting dots are too large (or small), you can reduce (or increase) dot sizes by adjusting this value.
    
    approx_clusternum : int
        The approximate number of row clusters to be created. Labeling the groups of leaves with different colors. The result of hierarchical clustering won't change.    
    approx_clusternum_col : int
        The approximate number of column clusters to be created. Labeling the groups of leaves with different colors. The result of hierarchical clustering won't change.
    
    color_var : int
        The title for color values. If not set, "color_val" will be used.
    merginalsum : bool
        Whether or not to draw bar plots for merginal distribution.
    show : bool
        Whether or not to show the figure.
    method : string
        Method for hierarchical clustering.
    return_col_cluster : string
        The title for color values. If not set, "color_val" will be used.
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
    """#print(kwargs)
    rnum, cnum=df.shape
    sns.set(font_scale=1)
    
    totalrowplot=0
    if merginalsum==True:
        totalrowplot+=1
    totalrowplot+=len(row_plot)
    totalrowplot+=len(row_colormap)
    totalcolplot=0
    if merginalsum==True:
        totalcolplot+=1
    totalcolplot+=len(col_plot) 
    totalcolplot+=len(col_colormap)
    
    
    if totalrowplot + totalcolplot >0:
        rowplotcount=0
        colplotcount=0
        row_colors=[]
        row_colors_title=[]
        col_colors=[]
        col_colors_title=[]
        
        if merginalsum:
            row_colors.append(np.ones([rnum, 4]))
            row_colors_title.append("Sum")
            col_colors.append(np.ones([rnum, 4]))
            col_colors_title.append("Sum")
        if len(row_colormap)>0:
            for k, v in row_colormap.items():
                row_colors.append(v)
                row_colors_title.append(k)
                
        if len(col_colormap)>0:
            for k, v in col_colormap.items():
                col_colors.append(v)
                col_colors_title.append(k)
        
        if len(row_plot)>0:
            for k, v in row_plot.items():
                row_colors.append(np.ones([rnum, 4]))
                row_colors_title.append(k)
        if len(col_plot)>0:
            for k, v in col_plot.items():
                col_colors.append(np.ones([rnum, 4]))
                col_colors_title.append(k)        
        
        
        
        
        
        if len(row_colors) >0 and len(col_colors) >0:
            g=sns.clustermap(df,col_colors=col_colors, row_colors=row_colors,method=method,**kwargs)
            g.ax_col_colors.invert_yaxis()
            g.ax_row_colors.invert_xaxis()
        elif len(col_colors) >0:
           
            g=sns.clustermap(df,col_colors=col_colors,method=method,**kwargs)
            g.ax_col_colors.invert_yaxis()
        elif len(row_colors) >0:
            g=sns.clustermap(df,row_colors=row_colors,method=method,**kwargs)
            g.ax_row_colors.invert_xaxis()
        
        rowplotcount=0
        colplotcount=0
        if merginalsum:
            mat=df.to_numpy()
            r=np.sum(mat, axis=1)
            g.ax_row_colors.barh(np.arange(r.shape[0])+0.5, r[leaves_list(g.dendrogram_row.linkage)]/np.amax(r))
            
            
            c=np.sum(mat, axis=0)
            #print(leaves_list(g.dendrogram_col.linkage))
            g.ax_col_colors.bar(np.arange(c.shape[0])+0.5,c[leaves_list(g.dendrogram_col.linkage)]/np.amax(c))
            
            rowplotcount=1
            colplotcount=1
        rowplotcount+=len(row_colormap)
        
        if len(row_plot)>0:
            row_cluster=True
            if "row_cluster" in kwargs:
                row_cluster=kwargs["row_cluster"]
            
            for i, (lname, r) in enumerate(row_plot.items()):
                r=np.array(r)
                if row_cluster==True:
                    tmpindx=leaves_list(g.dendrogram_row.linkage)
                    r=r[tmpindx]
                    r=r-np.amin(r)
                    r=r/np.amax(r)
                    r=0.9*r
                    g.ax_row_colors.plot(r+rowplotcount, np.arange(r.shape[0])+0.5)
                else:
                    g.ax_row_colors.plot(r/(np.amax(r)*1.1)+rowplotcount, np.arange(r.shape[0])+0.5)
            
                rowplotcount+=1
                
        
        colplotcount+=len(col_colormap)
        
        if len(col_plot)>0:
            col_cluster=True
            if "col_cluster" in kwargs:
                col_cluster=kwargs["col_cluster"]
            for i, (lname, r) in enumerate(col_plot.items()):
                r=np.array(r)
                if col_cluster==True:
                    g.ax_col_colors.plot(np.arange(r.shape[0])+0.5,r[leaves_list(g.dendrogram_col.linkage)]/(np.amax(r)*1.1)+colplotcount)
                else:
                    g.ax_col_colors.plot(np.arange(r.shape[0])+0.5,r/(np.amax(r)*1.1)+colplotcount)
                
                colplotcount+=1
        
        g.ax_row_colors.set_xticks(np.arange(len(row_colors_title))+0.5)
        g.ax_row_colors.set_xticklabels(row_colors_title, rotation=90)
        g.ax_col_colors.set_yticks(np.arange(len(col_colors_title))+0.5)
        g.ax_col_colors.set_yticklabels(col_colors_title)
        
        for title, colorlut in row_color_legend.items():
            legendhandles=[]
            for label, color in colorlut.items():
                legendhandles.append(Line2D([0], [0], color=color,linewidth=5, label=label))
            #g.add_legend(legend_data=legendhandles,title="Aroma",label_order=["W","F","Y"])
            g.ax_col_dendrogram.legend(handles=legendhandles, loc='upper right', title=title)
        for title, colorlut in col_color_legend.items():
            legendhandles=[]
            for label, color in colorlut.items():
                legendhandles.append(Line2D([0], [0], color=color,linewidth=5, label=label))
            #g.add_legend(legend_data=legendhandles,title="Aroma",label_order=["W","F","Y"])
            g.ax_col_dendrogram.legend(handles=legendhandles, loc='upper right', title=title)
        
        
    else:
        g=sns.clustermap(df,method=method,**kwargs)
    if color_var>0:
        cmap = cm.nipy_spectral(np.linspace(0, 1, color_var))
    else:
        cmap = cm.nipy_spectral(np.linspace(0, 1, approx_clusternum+5))
    hierarchy.set_link_color_palette([mpl.colors.rgb2hex(rgb[:3]) for rgb in cmap])
    
    """coloring the row dendrogram based on branch numbers crossed with the threshold"""
    if g.dendrogram_row != None:
        t=_dendrogram_threshold(g.dendrogram_row.dendrogram)
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
    t=_dendrogram_threshold(g.dendrogram_col.dendrogram)
    # lbranches=np.array(g.dendrogram_col.dendrogram["dcoord"])[:,:2]
    # rbranches=np.array(g.dendrogram_col.dendrogram["dcoord"])[:,2:]
    # thre=np.linspace(0, np.amax(g.dendrogram_col.dendrogram["dcoord"]), 100)[::-1]
    # for t in thre:
    #     #print(np.sum(lbranches[:,1]>t),np.sum(rbranches[:,0]>t),np.sum(lbranches[:,0]>t),np.sum(rbranches[:,1]>t))
    #     crossbranches=np.sum(lbranches[:,1]>t)+np.sum(rbranches[:,0]>t)-np.sum(lbranches[:,0]>t)-np.sum(rbranches[:,1]>t)
    #     #print(crossbranches)
    #
    #     if crossbranches>approx_clusternum_col:
    #         break
    
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
    
    
    
    if show:
        plt.show()
    else:
        if return_col_cluster==True:
            return pd.DataFrame(cdata), pd.DataFrame(col_cdata), g
        else:
            return pd.DataFrame(cdata), g


def triangle_heatmap(df, 
                     grid_pos: list=[],
                     grid_labels: list=[],
                     show: bool=False):
    
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
    if show:
        plt.show()
    return ax
    
    

def decomplot(df,category: str="", 
              method: str="pca", 
              component: int=3,
              arrow_color: str="yellow",
              arrow_text_color: str="black",
              show: bool=False, 
              explained_variance: bool=True,
              arrow_num: int=3,
              figsize=[]) :
    
    """
    Decomposing data and drawing a scatter plot and some plots for explained variables. 
    
    Parameters
    ----------
    df : pandas DataFrame
    category: str
        the column name of a known sample category (if exists). 
    method: str
        Method name for decomposition. Available methods: ["pca", "nmf"]
    component: int
        The component number
    
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
    if category !="":
        category_val=df[category].values
        df=df.drop([category], axis=1)
        x = df.values
        assert x.dtype==float, f"data must contain only float values except {category} column."
        
    else:    
        x = df.values
        assert x.dtype==float, "data must contain only float values."
        
    features=df.columns
    dfpc_list=[]
    if method=="pca":
        x=zscore(x, axis=0)
        pca = PCA(n_components=component, random_state=0)
        pccomp = pca.fit_transform(x)
        
        comb=list(combinations(np.arange(component), 2))
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
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        combnum=0
        for (i, j), ax in zip(comb, axes):
            xlabel, ylabel='pc'+str(i+1), 'pc'+str(j+1)
            dfpc = pd.DataFrame(data = np.array([pccomp[:,i],pccomp[:,j]]).T, columns = [xlabel, ylabel])
            if category!="":
                dfpc[category]=category_val
                if combnum==1:
                    sns.scatterplot(data=dfpc, x=xlabel, y=ylabel, hue=category, ax=ax)
                    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
                else:
                    sns.scatterplot(data=dfpc, x=xlabel, y=ylabel, hue=category, ax=ax,
                                    legend=False)
            else:
                sns.scatterplot(data=dfpc, x=xlabel, y=ylabel, ax=ax)
            _loadings=np.array([loadings[:,i],loadings[:,j]]).T
            a=np.sum(_loadings**2, axis=1)
            srtindx=np.argsort(a)[::-1][:arrow_num]
            _loadings=_loadings[srtindx]
            _features=np.array(features)[srtindx]
            for k, feature in enumerate(_features):
                
                #ax.plot([0,_loadings[k, 0] ], [0,_loadings[k, 1] ],color=arrow_color)
                ax.arrow(0, 0, _loadings[k, 0],_loadings[k, 1],color=arrow_color,width=0.005,head_width=0.1)
                ax.text(_loadings[k, 0],_loadings[k, 1],feature,color=arrow_text_color)
    
            dfpc_list.append(dfpc)
            combnum+=1
        plt.tight_layout(pad=0.5)
        if explained_variance==True:
            fig, ax=plt.subplots()
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
        
        if show==True:
            plt.show()
        else:
            return {"dataframe": dfpc_list,"pca_val": pccomp}
    elif method=="nmf":
        nmf=NMF(n_components=component)
        W = nmf.fit_transform(x)
        H = nmf.components_
        comb=list(combinations(np.arange(component), 2))
        fig, axes=plt.subplots(ncols=2, nrows=len(comb)//2+int(len(comb)%2!=0))
        axes=axes.flatten()
        for (i, j), ax in zip(comb, axes):
            xlabel, ylabel='p'+str(i+1), 'p'+str(j+1)
            dfpc = pd.DataFrame(data = np.array([W[:,i],W[:,j]]).T, columns = [xlabel, ylabel])
            dfpc[category]=category_val
            sns.scatterplot(data=dfpc, x=xlabel, y=ylabel, hue=category, ax=ax)
            dfpc_list.append(dfpc)
        
        fig.tight_layout()
        
        if explained_variance==True:
            fig, axes=plt.subplots(nrows=component, figsize=[5,5])
            axes=axes.flatten()
            for i, ax in enumerate(axes):
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
            
            
            
        if show==True:
            plt.show()
        return dfpc_list, W, H
    else:
        raise Exception('{} is not in options. Available options are: pca, nmf'.format(method))
def manifoldplot(df,category="", method="tsne",n_components=2,n_neighbors=4, show=False, **kwargs):
    """
    Reducing the dimensionality of data and drawing a scatter plot. 
    
    Parameters
    ----------
    df : pandas DataFrame
    category: str
        the column name of a known sample category (if exists). 
    method: str
        Method name for decomposition. 
        Available methods: ["tsne", 
                            "isomap",
                            "random_projection",
                            "linear_discriminant",
                            "lle",
                            "modlle",
                            "hessian_lle",
                            "mds"]
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
    
    if category !="":
        category_val=df[category].values
        df=df.drop([category], axis=1)
        x = df.values
        assert x.dtype==float, f"data must contain only float values except {category} column."
        
    else:    
        x = df.values
        assert x.dtype==float, "data must contain only float values."
    x=zscore(x, axis=0)
    features=df.columns
    
    if method=="random_projection": 
        embedding=SparseRandomProjection(
            n_components=n_components, random_state=42
        )
    elif method=="linear_discriminant": 
        embedding=LinearDiscriminantAnalysis(
            n_components=n_components
        )
    elif method=="isomap": 
        embedding=Isomap(n_neighbors=n_neighbors, n_components=n_components)
    
    elif method=="lle": 
        embedding=LocallyLinearEmbedding(
            n_neighbors=n_neighbors, n_components=n_components, method="standard"
        )
    elif method=="modlle": 
        embedding=LocallyLinearEmbedding(
            n_neighbors=n_neighbors, n_components=n_components, method="modified"
        )
    elif method=="hessian_lle": 
        embedding=LocallyLinearEmbedding(
            n_neighbors=n_neighbors, n_components=n_components, method="hessian"
        )
    elif method=="ltsa_lle": 
        embedding=LocallyLinearEmbedding(
            n_neighbors=n_neighbors, n_components=n_components, method="ltsa"
        )
    elif method=="mds": 
        embedding=MDS(
            n_components=n_components, n_init=1, max_iter=120, n_jobs=2, normalized_stress="auto"
        )
    elif method=="random_trees": 
        embedding=make_pipeline(
            RandomTreesEmbedding(n_estimators=200, max_depth=5, random_state=0),
            TruncatedSVD(n_components=n_components),
        )
    elif method=="spectral": 
        embedding=SpectralEmbedding(
            n_components=n_components, random_state=0, eigen_solver="arpack"
        )
    elif method=="tsne": 
        embedding=TSNE(
            n_components=n_components,
            n_iter=500,
            n_iter_without_progress=150,
            n_jobs=2,
            random_state=0,perplexity=10
        )
    elif method=="nca": 
        embedding=NeighborhoodComponentsAnalysis(
            n_components=n_components, init="pca", random_state=0
        )
    elif method=="umap": 
        embedding=umap.UMAP(
            min_dist=0.25,n_neighbors=15
        )
    else:
        raise Exception(f"Medthod {method} does not exist.")
    Xt=embedding.fit_transform(x)
    dft = pd.DataFrame(data = np.array([Xt[:,0],Xt[:,1]]).T, columns = ["d1", "d2"])
    if category !="":
        fig, ax=plt.subplots()
        dft[category]=category_val
        sns.scatterplot(data=dft, x="d1", y="d2", hue=category, ax=ax,**kwargs)
    else:
        sns.scatterplot(data=dft, x="d1", y="d2", ax=ax,**kwargs)
    if show==True:
        plt.show()
    return dft, ax

def clusterplot(df,category: str="", 
              method: str="kmeans",
              n_clusters: Union[str , int]=3,
              x: str="",
              y: str="",
              reduce_dimension: str="umap", 
              testrange=[1,20],
              show: bool=False,
              min_dist: float=0.25,
              n_neighbors: int=15,
              eps: Union[List[float], float]=0.5,
              pcacomponent: Optional[int]=None,
              ztranform=True,
              palette=["Spectral","cubehelix"]):
    """
    Clustering data and draw them as a scatter plot optionally with dimensionality reduction.  
    
    Parameters
    ----------
    df : pandas DataFrame
    
    category: str
        the column name of a known sample category (if exists). 
    method: str
        Method name for clustering. 
        Available methods: ["kmeans", 
                            "hierarchical",
                            "dbscan",
                            ]
    n_clusters: int or str
        The number of clusters to be created. If "auto" is provided, it will estimate optimal 
        cluster numbers with "Sum of squared distances" for k-mean clustering and silhouette method for others. 
    eps: int or list[int]
        DBSCAN's hyper parameter. It will affect the total number of clusters. 
    reduce_dimension: str
        Dimensionality reduction method. Default: umap. 
        if "" is passed, no reduction methods are applied. In this case, data must be 2 dimention or x and y options must be specified.
    
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
    
    
    
    
    if category !="":
        category_val=df[category].values
        df=df.drop([category], axis=1)
        X = df.values
        assert X.dtype==float, f"data must contain only float values except {category} column."
        
    else:    
        X = df.values
        assert X.dtype==float, "data must contain only float values."
    
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
            km = KMeans(n_clusters=k)
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
        
        plt.plot([K[srtindex[0]],K[srtindex[0]]],[0,np.amax(Sum_of_squared_distances)], "--", color="r")
        plt.text(K[srtindex[0]], np.amax(Sum_of_squared_distances)*0.95, "N="+str(K[srtindex[0]]))
        plt.plot([K[srtindex[1]],K[srtindex[1]]],[0,np.amax(Sum_of_squared_distances)], "--", color="r")
        plt.text(K[srtindex[1]], np.amax(Sum_of_squared_distances)*0.95, "N="+str(K[srtindex[1]]))
        plt.xticks(K)
        plt.xlabel('K')
        plt.ylabel('Sum of squared distances')
        plt.title('Elbow method for optimal cluster number')    
        
        print("Top two optimal No are: {}, {}".format(K[srtindex[0]],K[srtindex[1]]))
        n_clusters=[K[srtindex[0]],K[srtindex[1]]]
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
        print(scores)
        scores=np.array(scores)
        srtindex=np.argsort(scores)[::-1]
        plt.subplots()
        plt.plot(newK, scores, '-')
        plt.plot([newK[srtindex[0]],newK[srtindex[0]]],[0,np.amax(scores)], "--", color="r")
        plt.text(newK[srtindex[0]], np.amax(scores)*0.95, "N="+str(newK[srtindex[0]]))
        plt.plot([newK[srtindex[1]],newK[srtindex[1]]],[0,np.amax(scores)], "--", color="r")
        plt.text(newK[srtindex[1]], np.amax(scores)*0.95, "N="+str(newK[srtindex[1]]))
        plt.xticks(newK)
        plt.xlabel('K')
        plt.ylabel('Silhouette scores')
        plt.title('Optimal cluster number searches by silhouette method')    
        
        print("Top two optimal No are: {}, {}".format(newK[srtindex[0]],newK[srtindex[1]]))
        n_clusters=[newK[srtindex[0]],newK[srtindex[1]]]
        
    elif n_clusters=="auto" and method=="dbscan":
        # import scipy.spatial.distance as ssd
        # D=ssd.pdist(X)
        #
        from sklearn.neighbors import NearestNeighbors
        neigh = NearestNeighbors(n_neighbors=2)
        nbrs = neigh.fit(X)
        distances, indices = nbrs.kneighbors(X)
        distances = np.sort(distances[:,1], axis=0)
        #
        # plt.plot(distances)
        # plt.show()
        # sys.exit()
        print(np.amin(distances), np.amax(distances))
        K=np.linspace(np.amin(distances), np.amax(distances),20)
        newK=[]
        scores=[]
        _K=[]
        for k in K:
            db = DBSCAN(eps=k, min_samples=5, n_jobs=-1)
            dbX=db.fit(X)
            labels=np.unique(dbX.labels_[dbX.labels_>=0])
            print(k,labels)
            if len(labels)<2:
                continue
            _k=len(labels)
            if not _k in newK:
                newK.append(_k)
                _K.append(k)
                scores.append(silhouette_score(X[dbX.labels_>=0], dbX.labels_[dbX.labels_>=0], metric = 'euclidean')/_k)
        print(scores)
        scores=np.array(scores)
        
        _ksort=np.argsort(newK)
        _K=np.array(_K)[_ksort]
        newK=np.array(newK)[_ksort]
        scores=np.array(scores)[_ksort]
        srtindex=np.argsort(scores)[::-1]
        plt.subplots()
        plt.plot(newK, scores, '-')
        plt.plot([newK[srtindex[0]],newK[srtindex[0]]],[0,np.amax(scores)], "--", color="r")
        plt.text(newK[srtindex[0]], np.amax(scores)*0.95, "N="+str(newK[srtindex[0]]))
        plt.plot([newK[srtindex[1]],newK[srtindex[1]]],[0,np.amax(scores)], "--", color="r")
        plt.text(newK[srtindex[1]], np.amax(scores)*0.95, "N="+str(newK[srtindex[1]]))
        plt.xticks(newK)
        plt.xlabel('eps')
        plt.ylabel('Silhouette scores')
        plt.title('Optimal cluster number searches by silhouette method')    
        
        print("Top two optimal No are: {}, {}".format(newK[srtindex[0]],newK[srtindex[1]]))
        eps=[_K[srtindex[0]],_K[srtindex[1]]]
        
    else:
        n_clusters=[n_clusters]
    if method=="kmeans":
        dfnews=[]
        if reduce_dimension=="umap":
            x="UMAP1"
            y="UMAP2"
        for nc in n_clusters:
            kmean = KMeans(n_clusters=nc, random_state=0)
            kmX=kmean.fit(X)
            labels=np.unique(kmX.labels_)
            
            dfnew=pd.DataFrame(data = np.array([X[:,0],X[:,1]]).T, columns = [x, y])
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
                
            dfnew=pd.DataFrame(data = np.array([X[:,0],X[:,1]]).T, columns = [x, y])
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
            
            dfnew=pd.DataFrame(data = np.array([X[:,0],X[:,1]]).T, columns = [x, y])
            dfnew["dbscan"]=dbX.labels_
            dfnews.append(dfnew)
            tmp=0
            for c in set(dbX.labels_):
                if c >=0:
                    tmp+=1
            n_clusters.append(str(tmp)+", eps="+str(np.round(e,2)))
            
            
        hue="dbscan"
    _dfnews={}
    for dfnew, K in zip(dfnews, n_clusters): 
        if category=="":
            axnum=1
            fig, ax=plt.subplots(ncols=1, figsize=[4,4])
            ax=[ax]
        else:
            fig, ax=plt.subplots(ncols=2, figsize=[8,4])
        sns.scatterplot(data=dfnew,x=x,y=y,hue=hue, ax=ax[0], palette=palette[0])
        ax[0].set_title("Cluster number="+str(K))
        if category !="":
            dfnew[category]=category_val
            sns.scatterplot(data=dfnew,x=x,y=y,hue=category, ax=ax[1], palette=palette[1])
        _dfnews[K]=dfnew 
    return _dfnews

def volcanoplot():
    pass

def violinplot(df, 
               x: Optional[str]=None, 
               y: Optional[str]=None,
               pairs: list=[], 
               test: str="ttest_ind",
               alternative: str="two-sided",
               significance: str="numeric",
               significance_ranges: Dict[str, float]={"*":-np.log10(0.05),"**":4,"***":10},
               swarm: bool=False,
               xorder: list=[],
               equal_var: bool=False,**kwargs):
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
    dict("p values":pvalues,"ax":ax)
    
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
    
    return {"p values":newpvals,"ax":ax}
if __name__=="__main__":
    
    
    test="complex_clustermap"
    #test="dotplot"
    #test="triangle_heatmap"
    test="decomp"
    test="manifold"
    test="triangle_heatmap"
    test="radialtree"
    test="decomp"
    test="cluster"
    test="violinplot"
    if test=="dotplot":
        # df=pd.read_csv("/home/koh/ews/idr_revision/clustering_analysis/cellloc_pval_co.csv",index_col=0)
        # dfc=pd.read_csv("/home/koh/ews/idr_revision/clustering_analysis/cellloc_odds_co.csv",index_col=0)
        # from natsort import natsort_keygen
        # df=df.sort_index(axis=0,key=natsort_keygen())
        # dfc=dfc.sort_index(axis=0,key=natsort_keygen())
        # print(df)
        # dotplot(df, dfc=dfc,color_title="Odds ratio", size_title="-log10 p value",scaling=20)
        df=pd.read_csv("/home/koh/ews/idr_revision/clustering_analysis/cellloc_longform.csv")
        #df=pd.read_csv("/media/koh/grasnas/home/data/IDR/cluster16_go_summary.csv")
        print(df.columns)
        df=df.fillna(0)
        #dotplot(df, size_val="pval",color_val="odds", highlight="FDR",color_title="Odds ratio", size_title="-log10 p value",scaling=20)
        dotplot(df, row="Condensate",col="Cluster", size_val="pval",color_val="odds", highlight="FDR",
                color_title="Odds", size_title="-log10 p value",scaling=20,save="/media/koh/grasnas/home/data/IDR/cluster16_go_summary")
    elif test=="triangle_heatmap":
        s=20
        mat=np.arange(s*s).reshape([s,s])
        import string, random
        letters = string.ascii_letters+string.digits
        labels=[''.join(random.choice(letters) for i in range(10)) for _ in range(s)]
        df=pd.DataFrame(data=mat, index=labels, columns=labels)
        triangle_heatmap(df,grid_pos=[2*s//10,5*s//10,7*s//10],grid_labels=["A","B","C","D"])
    elif test=="complex_clustermap":
        _df=pd.DataFrame(np.arange(100).reshape([10,10]))
        cmap=plt.get_cmap("tab20b")
        complex_clustermap(_df,
                           row_colormap={"test1":[cmap(v) for v in np.linspace(0,1,10)]},
                           row_plot={"sine":np.sin(np.linspace(0,np.pi,10))},
                           approx_clusternum=3,
                           merginalsum=True)
    elif test=="radialtree":
        df=sns.load_dataset("penguins")
        df=df.dropna(axis=0)
        radialtree(df, category=["species","island","sex"])
        plt.show()
    elif test=="decomp":
        df=sns.load_dataset("penguins")
        df=df.dropna(axis=0)
        features=["species","bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]
        df=df[features]
        decomplot(df,category="species",method="pca")
        plt.show()
    elif test=="manifold":
        df=sns.load_dataset("penguins")
        df=df.dropna(axis=0)
        features=["species","bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]
        df=df[features]
        manifoldplot(df,category="species",method="tsne")
        plt.show()
    elif test=="cluster":
        df=sns.load_dataset("penguins")
        df=df.dropna(axis=0)
        features=["species","bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]
        df=df[features]
        clusterplot(df,category="species",method="hierarchical",n_clusters="auto")
        #clusterplot(df,category="species",method="dbscan",eps=0.35)
        plt.show()
    elif test=="violinplot":
        df=sns.load_dataset("penguins")
        df=df.dropna(axis=0)
        violinplot(df,x="species",y="bill_length_mm", 
                   pairs=[["Adelie","Chinstrap" ],["Gentoo","Chinstrap" ],["Adelie","Gentoo" ]],
                   test="mannwhitneyu",
                   significance="symbol",swarm=True)
        plt.show()
        
        