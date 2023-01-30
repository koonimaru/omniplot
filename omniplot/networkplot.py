import networkx as nx
import matplotlib.pyplot as plt
import igraph
from omniplot import igraph_classes
import numpy as np
from natsort import natsorted as nts
from matplotlib.lines import Line2D
import sys
import seaborn as sns
from typing import Union
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

def sankey():
    
    pass


if __name__=="__main__":
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
    