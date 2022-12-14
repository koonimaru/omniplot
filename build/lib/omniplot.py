"""
       /\   /\
      / \  / \
   @@@@@@@@@@@@@@
  @@@@@@@@@@@@@@@@
   (  ---  ---   )
   (   O L  O    )8
    (   <--->   )
     `---------'
     
"""
import matplotlib.collections as mc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import cm
import matplotlib as mpl

from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import leaves_list
from scipy.cluster import hierarchy
from collections import defaultdict
import matplotlib.colors
from natsort import natsort_keygen
from matplotlib.patches import Rectangle
import scipy.cluster.hierarchy as sch
import fastcluster as fcl
sns.set_theme()
def dotplot(df,
            row="",
            col="",
            dfc=pd.DataFrame(),
            scaling=10,
            color_val="",
            size_val="",
            highlight="",
            color_title="",
            size_title="",
            save="",
            threshold=-np.log10(0.05),
            row_clustering=True,
            xtickrotation=0):

    if size_val!="":
        _df=df.pivot_table(index=col,columns=row,values=size_val)
        _df=_df.sort_index(axis=0,key=natsort_keygen())
        _df=_df.fillna(0)
            
        if color_val!="":
            dfc=df.pivot_table(index=col,columns=row,values=color_val)
            dfc=dfc.sort_index(axis=0,key=natsort_keygen())
            dfc=dfc.fillna(0)
        if highlight !="":
            dfh=df.pivot_table(index=col,columns=row,values=highlight)
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
    ax1.margins(0.05)
    ax1.set_xticks(np.arange(len(_x)))
    ax1.set_xticklabels(_x,rotation=xtickrotation)
    ax1.set_yticks(np.arange(len(_y)))
    ax1.set_yticklabels(_y, rotation=0)
    if len(dfc) !=0:
        norm = mpl.colors.Normalize(vmin=np.min(cmat), vmax=np.amax(cmat))
        
        cb1 = mpl.colorbar.ColorbarBase(ax3, cmap=viridis,
                                        norm=norm,
                                        orientation='horizontal')
        cb1.set_label(color_title)
    #ax[1]=fig.add_axes([1,0.3,0.1,1])
    
    lxy=[[0.5, i*1] for i in range(3)]
    collection2 = mc.CircleCollection([middle0*scaling,middle1*scaling, maxsize*scaling], offsets=lxy, transOffset=ax2.transData, facecolors='gray',edgecolors="black")
    ax2.add_collection(collection2)
    ax2.axis('off')
    ax2.margins(0.3)
    for text, (x, y) in zip([middle0,middle1, maxsize], lxy):
        ax2.text(x*1.01, y,str(text),va="center")
    ax2.text(0.5,-0.5, size_title,va="center",ha="center")
    #ax[1].set_yticks(np.arange(3))
    #ax[1].set_yticklabels([minsize,middle, maxsize], rotation=0)
    #plt.tight_layout()
    if save!="":
        plt.savefig(save+".svg")
    plt.show()



colormap_list=["nipy_spectral", "terrain","gist_rainbow","CMRmap","coolwarm","gnuplot","gist_stern","brg","rainbow"]

def radialtree(Z2,fontsize=8,figsize=None, pallete="gist_rainbow", addlabels=True, show=True,sample_classes=None,colorlabels=None,
         colorlabels_legend=None):
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
    pallete : string
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
        figsize=[10,5]
    elif figsize==None and sample_classes != None:
        figsize=[10,5]
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
    plt.rcParams['font.family']= 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['svg.fonttype'] = 'none'
    xmax=np.amax(Z2['icoord'])
    ymax=np.amax(Z2['dcoord'])
    
    ucolors=sorted(set(Z2["color_list"]))
    #cmap = cm.gist_rainbow(np.linspace(0, 1, len(ucolors)))
    cmp=cm.get_cmap(pallete, len(ucolors))
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
        plt.plot([_xr0, _xr1], [_yr0, _yr1], c=_color,linewidth=linewidth)
        plt.plot([_xr2, _xr3], [_yr2,_yr3], c=_color,linewidth=linewidth)
        
        #plotting circular links between nodes
        if _yr1> 0 and _yr2>0:
            link=np.sqrt(r[1]**2-np.linspace(_xr1, _xr2, 100)**2)
            plt.plot(np.linspace(_xr1, _xr2, 100), link, c=_color,linewidth=linewidth)
        elif _yr1 <0 and _yr2 <0:
            link=-np.sqrt(r[1]**2-np.linspace(_xr1, _xr2, 100)**2)
            
            plt.plot(np.linspace(_xr1, _xr2, 100), link, c=_color,linewidth=linewidth)
        elif _yr1> 0 and _yr2 < 0:
            _r=r[1]
            if _xr1 <0 or _xr2 <0:
                _r=-_r
            link=np.sqrt(r[1]**2-np.linspace(_xr1, _r, 100)**2)
            plt.plot(np.linspace(_xr1, _r, 100), link, c=_color,linewidth=linewidth)
            link=-np.sqrt(r[1]**2-np.linspace(_r, _xr2, 100)**2)
            plt.plot(np.linspace(_r, _xr2, 100), link, c=_color,linewidth=linewidth)
        
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
            outerrad=outerrad-width*j-space*j
            innerrad=outerrad-width
            patches, texts =plt.pie(colorpos, colors=colorlist,
                    radius=outerrad,
                    counterclock=True,
                    startangle=label_coords[0][2]*0.5)
            circle=plt.Circle((0,0),innerrad, fc='whitesmoke')
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
            _cmp=cm.get_cmap(colormap_list[j], type_num)
            _colorlist=[_cmp(ucolors.index(c)/(type_num-1)) for c in colorlist]
            _colorlist=np.array(_colorlist)[Z2['leaves']]
            outerrad=outerrad-width*j-space*j
            innerrad=outerrad-width
            patches, texts =plt.pie(colorpos, colors=_colorlist,
                    radius=outerrad,
                    counterclock=True,
                    startangle=label_coords[0][2]*0.5)
            circle=plt.Circle((0,0),innerrad, fc='whitesmoke')
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
                       bbox_to_anchor=(1.5+0.3*i, 1.0),
                       title=labelname)
                plt.gca().add_artist(leg)
            #break
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.bottom.set_visible(False)
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
    if show==True:
        plt.show()
    else:
        return ax
    
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

def complex_clustermap(df,row_plot={},col_plot={},approx_clusternum=10,color_var=0,merginalsum=False,show=True,method="ward", **kwargs):
    print(kwargs)
    rnum, cnum=df.shape
    sns.set(font_scale=1)
    
    totalrowplot=0
    if merginalsum==True:
        totalrowplot+=1
    totalrowplot+=len(row_plot)

    totalcolplot=0
    if merginalsum==True:
        totalcolplot+=1
    totalcolplot+=len(col_plot)

    if totalrowplot + totalcolplot >0:
        rowplotcount=0
        colplotcount=0
        if totalrowplot >0 and totalcolplot >0:
            white_bgr=[np.ones([rnum, 4]) for _ in range(totalrowplot)]
            white_bgc=[np.ones([cnum, 4]) for _ in range(totalcolplot)]
            
            g=sns.clustermap(df,col_colors=white_bgc, row_colors=white_bgr,method=method,**kwargs)
            g.ax_col_colors.invert_yaxis()
            g.ax_row_colors.invert_xaxis()
        elif totalcolplot >0:
            white_bgc=[np.ones([cnum, 4]) for _ in range(totalcolplot)]
            
            g=sns.clustermap(df,col_colors=white_bgc,method=method,**kwargs)
            g.ax_col_colors.invert_yaxis()
        elif totalrowplot >0:
            white_bgr=[np.ones([rnum, 4]) for _ in range(totalrowplot)]
            
            g=sns.clustermap(df,row_colors=white_bgr,method=method,**kwargs)
            g.ax_row_colors.invert_xaxis()
        if merginalsum:
            mat=df.to_numpy()
            r=np.sum(mat, axis=1)
            g.ax_row_colors.barh(np.arange(r.shape[0])+0.5, r[leaves_list(g.dendrogram_row.linkage)]/np.amax(r))
            
            
            c=np.sum(mat, axis=0)
            #print(leaves_list(g.dendrogram_col.linkage))
            g.ax_col_colors.bar(np.arange(c.shape[0])+0.5,c[leaves_list(g.dendrogram_col.linkage)]/np.amax(c))
            
            rowplotcount=1
            colplotcount=1
        
        if len(row_plot)>0:
            row_cluster=True
            if "row_cluster" in kwargs:
                row_cluster=kwargs["row_cluster"]
            
            for i, (lname, r) in enumerate(row_plot.items()):
                r=np.array(r)
                if row_cluster==True:
                    g.ax_row_colors.plot(r[leaves_list(g.dendrogram_row.linkage)]/(np.amax(r)*1.1)+rowplotcount, np.arange(r.shape[0])+0.5)
                else:
                    g.ax_row_colors.plot(r/(np.amax(r)*1.1)+rowplotcount, np.arange(r.shape[0])+0.5)
                g.ax_row_colors.set_ylabel(lname)
                rowplotcount+=1
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
                g.ax_col_colors.set_ylabel(lname)
                colplotcount+=1
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


def triangle_heatmap(df, grid_pos=[],grid_labels=[]):
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
    plt.show()


if __name__=="__main__":
    
    
    test="radialtree"
    
    test="complex_clustermap"
    test="dotplot"
    #test="triangle_heatmap"
    
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
        df=pd.read_csv("/media/koh/grasnas/home/data/IDR/human_protein_atlas/rna_tissue_gtex.tsv", sep="\t")
        _df=df.pivot_table(index="Gene name",columns="Tissue",values="NX")
        _df=_df.loc[np.std(_df, axis=1)/np.mean(_df, axis=1)>=4]
        _df=pd.DataFrame(np.arange(1000).reshape([100,10]))
        complex_clustermap(_df,approx_clusternum=3,merginalsum=True)
    elif test=="radialtree":
        np.random.seed(1)
        numleaf=100
        _alphabets=[chr(i) for i in range(97, 97+24)]
        labels=sorted(["".join(list(np.random.choice(_alphabets, 10))) for i in range(numleaf)])
        x = np.random.rand(numleaf)
        D = np.zeros([numleaf,numleaf])
        for i in range(numleaf):
            for j in range(numleaf):
                D[i,j] = abs(x[i] - x[j])
        Y = sch.linkage(D, method='single')
        Z2 = sch.dendrogram(Y,labels=labels,no_plot=True)
        type_num=6
        type_list=["ex"+str(i) for i in range(type_num)]
        sample_classes={"example_color": [np.random.choice(type_list) for i in range(numleaf)]}
        radialtree(Z2, sample_classes=sample_classes)