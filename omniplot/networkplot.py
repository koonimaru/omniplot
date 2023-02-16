import networkx as nx
import matplotlib.pyplot as plt
import igraph
from omniplot import igraph_classes
import numpy as np
from natsort import natsorted as nts
from matplotlib.lines import Line2D
import sys
import seaborn as sns
from typing import Union, List, Dict, Optional
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import pandas as pd
from omniplot.chipseq_utils import _calc_pearson
from omniplot.utils import _baumkuchen_xy
from scipy.stats import zscore
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist, squareform
import itertools as it
from datashader.bundling import hammer_bundle
plt.rcParams['font.family']= 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['svg.fonttype'] = 'none'
sns.set_theme(font="Arial")

def pienodes(g,
             vertex_label: list=[], 
             node_features: dict={}, 
             pie_frac: str="frac",
             pie_label: str="label",
             pie_palette: Union[str , dict]="tab20b", 
             label_all: bool=True,
             piesize: float=0.1,
             **kwargs) -> plt.Axes: 
    """
    Drawing a network whose noses are pie charts.
    
    Parameters
    ----------
    g : igraph object
    vertex_label: list
        The list of node labels.
        e.g.: nodes=["A","B","C","D","E"]
    node_features: dict
        A dictionary containing fractions and labels of the pie charts.
        e.g.:
        pie_features={"A":{"frac":np.array([50,50]),"label":np.array(["a","b"])},
                  "B":{"frac":np.array([90,5,5]),"label":np.array(["a","b","c"])},
                  "C":{"frac":np.array([100]),"label":np.array(["c"])},
                  "D":{"frac":np.array([100]),"label":np.array(["b"])},
                  "E":{"frac":np.array([100]),"label":np.array(["a"])}}
    pie_frac : str
        The key value for the fractions of the pie charts. Default: "frac" (as the example of the above pie_features).
    pie_label : str
        The key value for the labels of the pie charts. Default: "label" (as the example of the above pie_features). 
    pie_palette: str or dict
        If string is provided, it must be one of the matplotlib colormap names for the pie charts. If dict, then  
    label_all: bool
        Whether to label all nodes or not. If False, labels show up only for 0.05 upper quantile of nodes with a high degree.
    
    piesize : float
        Scaling pie chart sizes if they are too large/small.    

    Returns
    -------
    axis
    
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
    
    
    
    if type(pie_palette)== str:
        colors={}
        unique_labels=set()
        for k, v in node_features.items():
            for la in v[pie_label]:
                unique_labels.add(la)
        cmap=plt.get_cmap(pie_palette)
        unique_labels=nts(unique_labels)
        labelnum=len(unique_labels)
        for i, ul in enumerate(unique_labels):
            colors[ul]=cmap(i/labelnum)
    elif type(pie_palette)==dict:
        colors=pie_palette
        unique_labels=nts(colors.keys())
    else:
        raise Exception("Unknown pie_palette type.")
    fig, ax = plt.subplots(figsize=[8,8])
    plt.subplots_adjust(right=0.8)
    mgd=igraph_classes.MatplotlibGraphDrawer(ax)
    mgd.draw(g,vertex_size=0.02,**kwargs)
    trans=ax.transData.transform
    trans2=fig.transFigure.inverted().transform
    
    deg=np.array([d for d in g.degree()])
    degsort=np.argsort(deg)
    nodes=np.array(vertex_label)
    deg=deg[degsort]
    nodes=nodes[degsort]
    pos=np.array(kwargs["layout"].coords)[degsort]
    deg=deg/deg.max()*piesize+0.015
    #piesize=0.02
    xycoord=[]
    text_pos=[]
    for i, n in enumerate(nodes):
        xx,yy=trans(pos[i]) # figure coordinates
        #print(xx,yy, pos[i])
        xa,ya=trans2((xx,yy)) # axes coordinates
        a = plt.axes([xa-deg[i]/2,ya-deg[i]/2, deg[i], deg[i]],
                     rasterized=True,
                     adjustable="datalim")
        #a.set_zorder(1)
        a.set_aspect('equal')
        
        a.pie(node_features[n][pie_frac], colors=[colors[f] for f in node_features[n][pie_label]])
        a.margins(0,0)
        
        if label_all==True:
            text_pos.append([a, xa,ya,n])
        else:
            if deg[i] > np.quantile(deg, 0.9):
                text_pos.append([a, xa,ya,n])
            #a.text(xa,ya,n)
        a.zorder=i
    for j, (a, xa,ya,n) in enumerate(text_pos):
        a.set_zorder(10)
        a.text(xa,ya,n)
        a.zorder=i+j
    legend_elements = [Line2D([0], [0], marker='o', color='lavender', label=ul,markerfacecolor=colors[ul], markersize=10)
                      for ul in unique_labels]
    
    ax.legend(handles=legend_elements,bbox_to_anchor=(0.95, 1))
    return ax

def sankey_category(df, 
                    category: list=[], 
                    palette: str="tab20c", 
                    colormode: str="independent",
                    altcat: str="",
                    show_percentage=False,
                    show_percentage_target=False,
                    fontsize: int=12) -> plt.Axes:
    #plt.rcParams.update({'font.size': 12})
    """
    Drawing a sankey plot to compare multiple categories in a data. The usage example may be 
    to compare between clustering results.
    
    Parameters
    ----------
    df : pandas dataframe
        it has to contain categorical columns specified by the 'category' option.
    category: list
        List of column names to compare.
    palette: str, optional
        Colormap name. Default: tab20c
    colormode: str ['shared', 'independent', 'alternative', 'trace'], optional
        The way to color categories. 'independent' will give each category a distinct (unrelated) colorset. 
        'shared' will give a shared color if categories share the same names of labels. 'alternative' will
        color bars based on additional category specified by altcat. 'trace' will color all bars according
        to the first category.  
        
    altcat : str, optional (but required when colormode='alternative')
        

    Returns
    -------
    axis
    
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
    if altcat !="":
        df = df.sort_values(category+[altcat])
    else:
        df = df.sort_values(category)
    df=df.reset_index(drop=True)
    
    #print(df)
    blockwidth=0.2
    xinterval=0.5
    space=0.02
    
    link_counts=[]
    catval=df[category].values
    catval=np.array(catval,dtype=str)
    for i in range(len(category)-1):
        s=category[i]
        t=category[i+1]
        sval=catval[:,i]
        tval=catval[:,i+1]
        links=[]
        for _sv, _tv in zip(sval,tval):
            links.append("-->>".join([str(_sv),str(_tv)]))
        ul, cl=np.unique(links, return_counts=True)
        link_counts.append([ul,cl])        
    
    heights={}
    xyh={}
    scolors={}
    for cat in category:
        u, c=np.unique(df[cat], return_counts=True)
        heights[cat]=[np.array(u,dtype=str),c]
    unique_cat=set()
    if colormode=="shared":
        for cat, v in heights.items():
            for c in list(v[0]):
                unique_cat.add(c)
        unique_cat=sorted(list(unique_cat))
        _tmp=plt.get_cmap(palette, len(unique_cat))
        _cmap={unique_cat[i]: _tmp(i) for i in range(len(unique_cat))}
        cmap={}
        for cat in category:
            cmap[cat]=_cmap
    elif colormode=="independent":
        cmap={}
        for cat, v in heights.items():
            _tmp=plt.get_cmap(palette, v[0].shape[0])
            cmap[cat]={v[0][i]: _tmp(i) for i in range(v[0].shape[0])}
    elif colormode=="trace":
        altcat=category[0]
        altcat_list=list(df[altcat])
        altcat_unique=list(np.unique(altcat_list))
        _tmp=plt.get_cmap(palette, len(altcat_unique))
        altcat_dict={}
        for a in altcat_unique:
            altcat_dict[a]=_tmp(altcat_unique.index(a))
        altcat_colors=[]
        for a in altcat_list:
            altcat_colors.append(_tmp(altcat_unique.index(a)))
    elif colormode=="alternative":
        if altcat=="":
            raise Exception("If colormode is 'alternative', altcat must be specified")
        altcat_list=list(df[altcat])
        altcat_unique=list(np.unique(altcat_list))
        _tmp=plt.get_cmap(palette, len(altcat_unique))
        altcat_dict={}
        for a in altcat_unique:
            altcat_dict[a]=_tmp(altcat_unique.index(a))
        altcat_colors=[]
        for a in altcat_list:
            altcat_colors.append(_tmp(altcat_unique.index(a)))
    
    fig, ax=plt.subplots(figsize=[2+len(category),7])
    blocks=[]
    facecolors=[]
    hs=[]
    for i,(cat, (u, ac)) in enumerate(heights.items()):
        xyh[cat]={}
        c=ac/np.sum(ac)
        h=0
        for _u, _c, _ac in zip(u,c,ac):
            blocks.append(Rectangle([i*xinterval, h],blockwidth,_c))
            if colormode=="alternative" or colormode=="trace":
                facecolors.append([1,1,1,1])
            else:
                facecolors.append(cmap[cat][_u])
            h+=_c+space
            xyh[cat][str(_u)]=[i*xinterval, h, _c,_ac]
            ax.text(i*xinterval+blockwidth/2,h-space-_c/2, _u,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="b", lw=1, alpha=0.7))
        hs.append(h)
    if colormode=="alternative" or colormode=="trace":
        
        for i,(cat, (u, ac)) in enumerate(heights.items()):
            k=0
            for j, _u in enumerate(u):
                dfcat=np.array(df[cat], dtype=str)
                _df=df.loc[dfcat==_u]
                _df=_df.sort_values(altcat)                
                x, h, _c, _ac=xyh[cat][_u]
                for l, label in enumerate(_df[altcat]):
                    plt.plot([x,x+blockwidth],[h-_c-space+_c*l/_ac,h-_c-space+_c*l/_ac], color=altcat_dict[label])
                    k+=1
        plt.legend([Line2D([0], [0], color=altcat_dict[label]) for label in altcat_dict.keys()],
                   altcat_dict.keys(),loc=[1.01,0.9], fontsize=fontsize)
        plt.subplots_adjust(right=0.7)
        
    # draw links between categories
    resolution=100 # resolution of lines for the links
    for i, (ul, cl) in enumerate(link_counts):
        scat=category[i]
        tcat=category[i+1]
        sbottom={}
        tbottom={}
        scats=set()
        tcats=set()
        for _ul, _cl in zip(ul, cl):
            
            
            s, t=_ul.split("-->>")
            sx, sy, sh, _=xyh[scat][s]
            tx, ty, th, _=xyh[tcat][t]

            if not t in tbottom:
                tbottom[t]=ty-space-th
            if not s in sbottom:
                sbottom[s]=sy-space-sh
            _scl=sh*_cl/(heights[scat][1][heights[scat][0]==s])[0]
            _tcl=th*_cl/(heights[tcat][1][heights[tcat][0]==t])[0]
            
            # morphing links by convolution
            xconv=np.linspace(sx+blockwidth, tx,resolution-20*2+2)
            byconv=np.array((resolution//2) * [sbottom[s]] + (resolution//2) * [tbottom[t]])
            byconv = np.convolve(byconv, 0.05 * np.ones(20), mode='valid')
            byconv = np.convolve(byconv, 0.05 * np.ones(20), mode='valid')
            #ax.plot(xconv,yconv)
            tyconv=np.array((resolution//2) * [_scl+sbottom[s]] + (resolution//2) * [tbottom[t]+_tcl])
            tyconv = np.convolve(tyconv, 0.05 * np.ones(20), mode='valid')
            tyconv = np.convolve(tyconv, 0.05 * np.ones(20), mode='valid')
            
            #ax.plot(xconv,yconv)
            plt.fill_between(
                xconv, byconv, tyconv, alpha=0.65,
                color="b"
            )
            
            if show_percentage==True:
                plt.text(sx, sh/2+sy-space-sh,str(np.round(100*sh,1))+"%",ha="right",
                         va="center",
                         rotation=90,
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="y", lw=1, alpha=0.8))
                if i==len(link_counts)-1:
                    plt.text(tx, th/2+ty-space-th,str(np.round(100*th,1))+"%",ha="right",
                             va="center",
                             rotation=90,
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="y", lw=1, alpha=0.8))
            if show_percentage_target==True:
                plt.text(sx+blockwidth, _scl/2+sbottom[s],
                         str(np.round(100*_scl,1))+"%",
                         ha="left",va="center",
                         rotation=90,
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="y", lw=1, alpha=0.8))
                
            sbottom[s]+=_scl
            tbottom[t]+=_tcl
            
    # Create patch collection with specified colour/alpha

    pc = PatchCollection(blocks,facecolor=facecolors,edgecolor="black",linewidth=2)
    

    # Add collection to axes
    ax.add_collection(pc)
    
    
    
    plt.xlim(-blockwidth*0.5, xinterval*(len(heights)-1)+blockwidth*1.5)
    plt.ylim(-0.01, np.amax(hs)*1.1)
    plt.xticks([i*xinterval+blockwidth/2 for i in range(len(category))],category, rotation=90, 
               fontsize=fontsize)
    plt.yticks([])
    plt.subplots_adjust(bottom=0.2)
    return ax


def _pienodes(g,
             vertex_label: list=[], 
             node_features: dict={}, 
             pie_frac: str="frac",
             pie_label: str="label",
             pie_palette: Union[str , dict]="tab20b", 
             label_all: bool=True,
             piesize: float=0.1,label_color="w",
             **kwargs) -> plt.Axes: 
    """
    Drawing a network whose noses are pie charts.
    
    Parameters
    ----------
    g : igraph object
    vertex_label: list
        The list of node labels.
        e.g.: nodes=["A","B","C","D","E"]
    node_features: dict
        A dictionary containing fractions and labels of the pie charts.
        e.g.:
        pie_features={"A":{"frac":np.array([50,50]),"label":np.array(["a","b"])},
                  "B":{"frac":np.array([90,5,5]),"label":np.array(["a","b","c"])},
                  "C":{"frac":np.array([100]),"label":np.array(["c"])},
                  "D":{"frac":np.array([100]),"label":np.array(["b"])},
                  "E":{"frac":np.array([100]),"label":np.array(["a"])}}
    pie_frac : str
        The key value for the fractions of the pie charts. Default: "frac" (as the example of the above pie_features).
    pie_label : str
        The key value for the labels of the pie charts. Default: "label" (as the example of the above pie_features). 
    pie_palette: str or dict
        If string is provided, it must be one of the matplotlib colormap names for the pie charts. If dict, then  
    label_all: bool
        Whether to label all nodes or not. If False, labels show up only for 0.05 upper quantile of nodes with a high degree.
    
    piesize : float
        Scaling pie chart sizes if they are too large/small.    

    Returns
    -------
    axis
    
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
    
    
    
    if type(pie_palette)== str:
        colors={}
        unique_labels=set()
        for k, v in node_features.items():
            for la in v[pie_label]:
                unique_labels.add(la)
        cmap=plt.get_cmap(pie_palette)
        unique_labels=nts(unique_labels)
        labelnum=len(unique_labels)
        for i, ul in enumerate(unique_labels):
            colors[ul]=cmap(i/labelnum)
    elif type(pie_palette)==dict:
        colors=pie_palette
        unique_labels=nts(colors.keys())
    else:
        raise Exception("Unknown pie_palette type.")
    
    pos=np.array(kwargs["layout"].coords)
    posmax=np.amax(pos)
    fig, ax = plt.subplots(figsize=[6,6*np.amax(pos[:,1])/np.amax(pos[:,0])])
    plt.subplots_adjust(right=0.8)
    mgd=igraph_classes.MatplotlibGraphDrawer(ax)
    
    _deg=np.array([d for d in g.degree()])
    _deg=np.log(_deg+2)
    _deg=0.05*posmax*(_deg/_deg.max())
    #print(_deg)
    degsort=np.argsort(_deg)
    nodes=np.array(vertex_label)
    deg=_deg[degsort]
    nodes=nodes[degsort]
    pos=np.array(kwargs["layout"].coords)[degsort]
    
    mgd.draw(g, vertex_size=1.8*_deg,alpha=0,**kwargs)
    #piesize=0.02
    #print(nodes)
    #print(deg)
    xycoord=[]
    text_pos=[]
    for i, n in enumerate(nodes):
        
        
        x, y=pos[i]
        frac=node_features[n][pie_frac]
        frac=2*np.pi*np.array(frac)/np.sum(frac)
        
        _colors=[colors[f] for f in node_features[n][pie_label]]
        angle=0
        for fr, co in zip(frac, _colors):
            _baumkuchen_xy(ax, x, y, angle, fr, 0, deg[i],20, co)
            angle+=fr
        # xx,yy=trans(pos[i]) # figure coordinates
        # #print(xx,yy, pos[i])
        # xa,ya=trans2((xx,yy)) # axes coordinates
        # a = plt.axes([xa-deg[i]/2,ya-deg[i]/2, deg[i], deg[i]],
        #              rasterized=True,
        #              adjustable="datalim")
        # #a.set_zorder(1)
        # a.set_aspect('equal')
        #
        # a.pie(node_features[n][pie_frac], colors=[colors[f] for f in node_features[n][pie_label]])
        # a.margins(0,0)
        
        if label_all==True:
            text_pos.append([x, y,n])
        else:
            if deg[i] > np.quantile(deg, 0.9):
                text_pos.append([x, y,n])

    for j, (xa,ya,n) in enumerate(text_pos):
        ax.text(xa,ya,n, color=label_color)
        ax.zorder=i+j
    legend_elements = [Line2D([0], [0], marker='o', color='lavender', label=ul,markerfacecolor=colors[ul], markersize=10)
                      for ul in unique_labels]
    
    ax.legend(handles=legend_elements,bbox_to_anchor=(0.95, 1))
    return {"axes":ax}


def sankey_flow(df):
    
    pass

def _bundlle_edges(G, pos):
    #print(G.nodes)
    nodes_to_int={}
    i=0
    for n in G.nodes:
        nodes_to_int[n]=i
        i+=1
    
    
    nodes_py = [[nodes_to_int[name], pos[name][0], pos[name][1]] for name in G.nodes]
    ds_nodes = pd.DataFrame(nodes_py, columns=['name', 'x', 'y'])       
    
    ds_edges_py = [[nodes_to_int[n0], nodes_to_int[n1]] for (n0, n1) in G.edges]
    ds_edges = pd.DataFrame(ds_edges_py, columns=['source', 'target'])
    
    hb = hammer_bundle(ds_nodes, ds_edges)
    return hb
    

def correlation(df: pd.DataFrame, 
                category: Union[str, list]=[],
                method="pearson",
                layout: str="spring_layout",
                palette: str="tab20c",
                clustering: str="",
                figsize: list=[],
                ztransform: bool=True,
                threshold: Union[float, str]=0.5,
                layout_param: dict={},
                node_edge_color: str="black",
                edge_color: str="weight",
                edge_cmap: str="hot",
                edge_width: Union[str, float]="weight",
                node_size: float=50,
                node_alpha: float=0.85,
                linewidths: float=0.5,
                n_jobs: int=-1,
                edges_alpha: float=0.7,
                edge_width_scaling: float=4,
                rows_cols: list=[],
                node_color="b",
                bundle: bool=True,
                show_edges: bool=True) -> Dict:
    """
    Drawing a network based on correlations or distances between observations.
    Parameters
    
    ----------
    df : pandas DataFrame
        
    category: str or list, optional (default: [])
        the names of categorical values to display as color labels
    method: str, optional (default: "pearson")
        method for correlation/distance calculation. Availables: "pearson", "euclidean", "cosine",
        if a distance method is chosen, sigmoidal function is used to convert distances into edge weights.
    layout: str, optional
        Networkx layouts include: pydot_layout, spring_layout, random_layout, circular_layout and so on. Please see https://networkx.org/documentation/stable/reference/drawing.html
    palette : str, optional (default: "tab20c")
        A colormap name.
    clustering: str, optional (default: "")
        Networkx clustering methods include:  best_partition
    figsize : List[int], optional
        The figure size, e.g., [4, 6].
    ztransform : bool, optional
        Whether to transform values to z-score
    threshold : float, optional (default: 0.5)
        A cutoff value to remove edges of which correlations/distances are less than this value
    layout_param: dict, optional
        Networkx layout parameters related to the layout option
    node_edge_color: str, optional (default: "black")
        The colors of node edges.
        
    edge_color: str, optional (default: "weight")
        The color of edges. The default will color edges based on the edge weights calculated based on pearson/distance methods.
    edge_cmap: str, optional (default: "hot")
        
    edge_width: Union[str, float]="weight",
    node_size: float=50,
    node_alpha: float=0.85,
    linewidths: float=0.5,
    n_jobs: int=-1,
    edges_alpha: float=0.7,
    edge_width_scaling: float=4,
    rows_cols: list=[],
    node_color="b",
    bundle: bool=True,
    show_edges: bool=True
    
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
    
    original_index=list(df.index)
    if len(category) !=0:

        if type(category)==str:
            category=[category]
        #df=df.drop(category, axis=1)
        valnames=list(set(df.columns) -set(category)) 
        X = df[valnames].values
        assert X.dtype==float, f"data must contain only float values except {category} columns."
        
    else:    
        X = df.values
        assert X.dtype==float, "data must contain only float values."
    if ztransform==True:
        X=zscore(X, axis=0)
    if method=="pearson":
        dmat=Parallel(n_jobs=n_jobs)(delayed(_calc_pearson)(ind, X) for ind in list(it.combinations(range(X.shape[0]), 2)))
        dmat=np.array(dmat)
        dmat=squareform(dmat)
        #print(dmat)
        dmat+=np.identity(dmat.shape[0])
    else:
        dmat=squareform(pdist(X, method))
        dmat=(dmat-np.mean(dmat))/np.std(dmat)
        #dmat=dmat/np.amax(dmat)
        dmat=(1+np.exp(-dmat))**-1
        #dmat=1/(1+dmat)
    G = nx.Graph()
    for index in original_index:
        G.add_node(index)
    for i, j in list(it.combinations(range(dmat.shape[0]), 2)):
        if dmat[i,j]>=threshold:
            G.add_edge(original_index[i], original_index[j], weight=dmat[i,j])
    
    if clustering =="best_partition": 
        try:
            import community
        except ImportError as e:
            raise("can not import community. Try 'pip install louvain'")
        comm=community.best_partition(G)
    
    elif clustering =="louvain":
        from networkx.algorithms import community
        comm=community.louvain.louvain_communities(G)
        
        
        
        #print(comm)
    weights=[]
    for s, t, w in G.edges(data=True):
        weights.append(w['weight'])
    weights=np.array(weights)
    
    #pos = nx.spring_layout(G, weight = 'weight', **spring_layout_param)
    layoutfunction = getattr(nx, layout)
    if layout=="spring_layout" and len(layout_param)==0:
        layout_param=dict(k=0.75,
                seed=0,
                scale=1,weight = 'weight')
        pos = layoutfunction(G,  **layout_param)
    else:
        pos = layoutfunction(G, **layout_param)
    #print(pos)
    colors={}
    colorlut={}
    for cat in category:
        _cats=df[cat]
        u=list(set(_cats))
        _cmp=plt.get_cmap(palette, len(u))
        _cmap_dict={k: _cmp(i) for i, k in enumerate(u)}
        colorlut[cat]=_cmap_dict
        colors[cat]=[]
        for g in G.nodes:
            colors[cat].append(_cmap_dict[_cats[original_index.index(g)]])
    if clustering =="best_partition":
        u=set()
        for k, v in comm.items():
            u.add(v)
        _cmp=plt.get_cmap(palette, len(u))
        _cmap_dict={k: _cmp(i) for i, k in enumerate(u)}
        colorlut[clustering]=_cmap_dict
        colors[clustering]=[]
        for g in G.nodes:
            colors[clustering].append(_cmap_dict[comm[g]])
    
    
    if edge_color=="weight":
        edge_color=weights
    if edge_width=="weight":
        edge_width=edge_width_scaling*weights


    if len(category)==0 and clustering=="":
        fig, ax=plt.subplots()
        nx.draw_networkx_nodes(G = G, node_color=node_color, 
                                   node_size=node_size,
                                   pos=pos,linewidths=linewidths, 
                                   ax=ax,edgecolors=node_edge_color, 
                                   alpha=node_alpha)
        nx.draw_networkx_edges(G = G,pos=pos, edge_color=edge_color, edge_cmap=plt.get_cmap(edge_cmap),
                               alpha=edges_alpha, 
                               width=edge_width, ax=ax)
    else:
        if len(figsize)==0:
            figsize=[4*(len(cat)+int(clustering!="")),4]
        if len(rows_cols)==0:
            rows_cols=[1, len(cat)+int(clustering!="")]
        fig, axes=plt.subplots(figsize=figsize, ncols=rows_cols[1], nrows=rows_cols[0])
        
        axes=axes.flatten()
        
        if bundle==True:
            hb=_bundlle_edges(G, pos)
        
        
        for ax, cat in zip(axes, category):
            #nx.draw(G=G,pos=pos, font_size=8,linewidths=0)
            nx.draw_networkx_nodes(G = G, node_color=colors[cat], 
                                   node_size=node_size,
                                   pos=pos,linewidths=linewidths, 
                                   ax=ax,edgecolors=node_edge_color, 
                                   alpha=node_alpha)
            if show_edges==True:
                nx.draw_networkx_edges(G = G,pos=pos, edge_color=edge_color, edge_cmap=plt.get_cmap(edge_cmap),
                                   alpha=edges_alpha, 
                                   width=edge_width, ax=ax)
                                    #,arrows=True,connectionstyle="arc3,rad=0.3")
            ax.set_title(method+", colored by "+cat)
            legendhandles=[]
            for label, color in colorlut[cat].items():
                legendhandles.append(Line2D([0], [0], color=color,linewidth=5, label=label))
            #g.add_legend(legend_data=legendhandles,title="Aroma",label_order=["W","F","Y"])
            ax.legend(handles=legendhandles, loc='best', title=cat)
            if bundle==True:
                ax.plot(hb.x, hb.y, "y", zorder=1, linewidth=3)
        if clustering!="":
            nx.draw_networkx_nodes(G = G, node_color=colors[clustering], 
                                   node_size=node_size,
                                   pos=pos,linewidths=linewidths, 
                                   ax=axes[-1],edgecolors=node_edge_color, 
                                   alpha=node_alpha)
            if show_edges==True:
                nx.draw_networkx_edges(G = G,pos=pos, edge_color=edge_color, edge_cmap=plt.get_cmap(edge_cmap),
                                   alpha=edges_alpha, 
                                   width=edge_width, ax=axes[-1])
                                    #,arrows=True,connectionstyle="arc3,rad=0.3")
            axes[-1].set_title(method+", colored by "+clustering)
            legendhandles=[]
            for label, color in colorlut[clustering].items():
                legendhandles.append(Line2D([0], [0], color=color,linewidth=5, label=label))
            axes[-1].legend(handles=legendhandles, loc='best', title=clustering)
            if bundle==True:
                axes[-1].plot(hb.x, hb.y, "y", zorder=1, linewidth=3)
    return {"axes":axes,"networkx":G, "distance_mat":dmat}


if __name__=="__main__":
    
    test="sankey_category"
    test="pienode"
    if test=="correlation":
        df=sns.load_dataset("penguins")
        df=df.dropna(axis=0)
        df=df.reset_index()
        correlation(df, category=["species", "island","sex"], 
                    method="pearson", 
                    ztransform=True,
                    clustering ="best_partition",show_edges=True, bundle=False)
        plt.show()
        
    elif test=="sankey_category":
        df=pd.read_csv("../data/kmeans_result.csv")
        sankey_category(df, ["kmeans2","kmeans3","sex"],
                        colormode="alternative",
                        altcat="species",
                        show_percentage=False,
                        show_percentage_target=False)
        plt.show()
    elif test=="pienode":
        edges=[[0,0],[0,1],[0,2],[2,1],[2,3],[3,4],[0,5]]
        edge_width=[1 for i in range(len(edges))]
        nodes=["A","B","C","D","E","F"]
        pie_features={"A":{"frac":np.array([50,50]),"label":np.array(["a","b"])},
                      "B":{"frac":np.array([90,5,5]),"label":np.array(["a","b","c"])},
                      "C":{"frac":np.array([100]),"label":np.array(["c"])},
                      "D":{"frac":np.array([100]),"label":np.array(["b"])},
                      "E":{"frac":np.array([100]),"label":np.array(["a"])},
                      "F":{"frac":np.array([10,20,30]),"label":np.array(["a","b","c"])}}
        
        g=igraph.Graph(edges=edges)
        layout = g.layout("fr")
        
        
        _pienodes(g, vertex_label=nodes,
                 node_features=pie_features,
                 layout=layout,
        vertex_color="lightblue",
        edge_color="gray",
        edge_arrow_size=0.03,
        edge_width=edge_width,
        keep_aspect_ratio=True)
        plt.show()
        