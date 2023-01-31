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
sns.set_theme()
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
                    altcat: str="") -> plt.Axes:
    
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
        color bars based on additional category specified by altcat.  
        
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
    
    print(df)
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
        cat=category[0]
        v=heights[cat]
        _tmp=plt.get_cmap(palette, v[0].shape[0])
        cmap={cat: {v[0][i]:_tmp(i) for i in range(v[0].shape[0])}}
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
    for i,(cat, (u, ac)) in enumerate(heights.items()):
        xyh[cat]={}
        c=ac/np.sum(ac)
        h=0
        for _u, _c, _ac in zip(u,c,ac):
            blocks.append(Rectangle([i*xinterval, h],blockwidth,_c))
            if colormode=="trace" and i >0:
                facecolors.append([1,1,1,1])
            elif colormode=="alternative":
                facecolors.append([1,1,1,1])
            else:
                facecolors.append(cmap[cat][_u])
            h+=_c+space
            xyh[cat][str(_u)]=[i*xinterval, h, _c,_ac]
            ax.text(i*xinterval+blockwidth/2,h-space-_c/2, _u,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="b", lw=1, alpha=0.7))
    
    if colormode=="alternative":
        
        for i,(cat, (u, ac)) in enumerate(heights.items()):
            k=0
            for j, _u in enumerate(u):
                dfcat=np.array(df[cat], dtype=str)
                _df=df.loc[dfcat==_u]                
                x, h, _c, _ac=xyh[cat][_u]
                for l, label in enumerate(_df[altcat]):
                    plt.plot([x,x+blockwidth],[h-_c-space+_c*l/_ac,h-_c-space+_c*l/_ac], color=altcat_dict[label])
                    k+=1
        plt.legend([Line2D([0], [0], color=altcat_dict[label]) for label in altcat_dict.keys()],
                   altcat_dict.keys(),loc=[1.01,0.9])
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
            
            
            sbottom[s]+=_scl
            tbottom[t]+=_tcl
            
    # Create patch collection with specified colour/alpha

    pc = PatchCollection(blocks,facecolor=facecolors,edgecolor="black",linewidth=2)
    

    # Add collection to axes
    ax.add_collection(pc)
    
    
    
    plt.xlim(-blockwidth*0.5, xinterval*(len(heights)-1)+blockwidth*1.5)
    plt.ylim(-0.01, 1.1)
    plt.xticks([i*xinterval+blockwidth/2 for i in range(len(category))],category, rotation=90)
    plt.subplots_adjust(bottom=0.2)
    return ax
    
def sankey_flow(df):
    
    pass


if __name__=="__main__":
    
    test="sankey_category"
    if test=="sankey_category":
        df=pd.read_csv("../data/kmeans_result.csv")
        sankey_category(df, ["kmeans2","kmeans3","sex"],colormode="alternative",altcat="species")
        plt.show()
    elif test=="pienode":
        edges=[[0,0],[0,1],[0,2],[2,1],[2,3],[3,4]]
        edge_width=[1 for i in range(len(edges))]
        nodes=["A","B","C","D","E"]
        pie_features={"A":{"frac":np.array([50,50]),"label":np.array(["a","b"])},
                      "B":{"frac":np.array([90,5,5]),"label":np.array(["a","b","c"])},
                      "C":{"frac":np.array([100]),"label":np.array(["c"])},
                      "D":{"frac":np.array([100]),"label":np.array(["b"])},
                      "E":{"frac":np.array([100]),"label":np.array(["a"])}}
        
        g=igraph.Graph(edges=edges)
        layout = g.layout("fr")
        
        
        pienodes(g, vertex_label=nodes,
                 node_features=pie_features,
                 layout=layout,
        vertex_color="lightblue",
        edge_color="gray",
        edge_arrow_size=0.03,
        edge_width=edge_width,
        keep_aspect_ratio=True)
        plt.show()
        