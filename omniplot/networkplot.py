import networkx as nx
import matplotlib.pyplot as plt
import igraph
from omniplot import igraph_classes
import numpy as np
from natsort import natsorted as nts
from matplotlib.lines import Line2D
import sys
import seaborn as sns
sns.set_theme()
def pienodes(g,
             vertex_label=[], 
             node_features={}, 
             pie_frac="frac",
             pie_label="label",
             pie_palette="tab20b", 
             label_all=True,
             piesize=0.1,
             **kwargs):
    if type(pie_label)== str:
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
    elif type(pie_label)==dict:
        colors=pie_palette
    else:
        sys.exit("Unknown pie_palette type.")
    fig, ax = plt.subplots(figsize=[8,8])
    
    mgd=igraph_classes.MatplotlibGraphDrawer(ax)
    mgd.draw(g,vertex_size=0.01,**kwargs)
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
                     zorder=0,
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
    for a, xa,ya,n in text_pos:
        a.set_zorder(10)
        a.text(xa,ya,n)
    
    legend_elements = [Line2D([0], [0], marker='o', color='lavender', label=ul,markerfacecolor=colors[ul], markersize=10)
                      for ul in unique_labels]
    
    ax.legend(handles=legend_elements,bbox_to_anchor=(1.1, 1))
    return ax

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
    