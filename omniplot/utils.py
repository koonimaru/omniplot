
import numpy as np 
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib
from typing import Optional, List, Dict, Union
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import leaves_list
from matplotlib import cm
from scipy.cluster import hierarchy
colormap_list=["nipy_spectral", "terrain","tab20b","gist_rainbow","CMRmap","coolwarm","gnuplot","gist_stern","brg","rainbow"]
plt.rcParams['font.family']= 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['svg.fonttype'] = 'none'
sns.set_theme()
def baumkuchen(ax, start, theta, rin, rout,res, _color):
    move=np.linspace(0, theta,res)
    xf=np.concatenate([rin*np.cos(start+move),[rin*np.cos(start+theta),rout*np.cos(start+theta)],rout*np.cos(start+move)[::-1],[rin*np.cos(start),rout*np.cos(start)][::-1]])
    yf=np.concatenate([rin*np.sin(start+move),[rin*np.sin(start+theta),rout*np.sin(start+theta)],rout*np.sin(start+move)[::-1],[rin*np.sin(start),rout*np.sin(start)][::-1]])
    ax.fill(xf, yf, color=_color)
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
    cmp=plt.get_cmap(palette, len(ucolors))
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
            _cmp=plt.get_cmap(colormap_list[j])
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


def _radialtree2(Z2,fontsize: int=8,
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
        figsize=[7,6]
    elif figsize==None and sample_classes != None:
        figsize=[7,6]
    elif figsize==None :
        figsize=[5,5]
    linewidth=0.5
    R=1
    width=R*0.1
    space=R*0.05
    if colorlabels != None:
        offset=width*len(colorlabels)/R+space*(len(colorlabels)-1)/R+0.05
        #print(offset)
    elif sample_classes != None:
        offset=width*len(sample_classes)/R+space*(len(sample_classes)-1)/R+0.05
        #print(offset)
    else:
        offset=0
    
    xmax=np.amax(Z2['icoord'])
    ymax=np.amax(Z2['dcoord'])
    
    ucolors=sorted(set(Z2["color_list"]))
    #cmap = cm.gist_rainbow(np.linspace(0, 1, len(ucolors)))
    cmp=plt.get_cmap(palette, len(ucolors))
    #print(cmp)
    if type(cmp) == matplotlib.colors.LinearSegmentedColormap:
        cmap = cmp(np.linspace(0, 1, len(ucolors)))
    else:
        cmap=cmp.colors
    fig, ax=plt.subplots(figsize=figsize)
    i=0
    label_coords=[]
    _lineres=1000
    for x, y, c in sorted(zip(Z2['icoord'], Z2['dcoord'],Z2["color_list"])):
    #x, y = Z2['icoord'][0], Z2['dcoord'][0]
        _color=cmap[ucolors.index(c)]
        if c=="C0": #np.abs(_xr1)<0.000000001 and np.abs(_yr1) <0.000000001:
            _color="black"
        
        # transforming original x coordinates into relative circumference positions and y into radius
        # the rightmost leaf is going to [1, 0]
        xinterval=np.array([x[0],x[2]])/xmax
        r=R*(1-np.array(y)/ymax)
        _x=np.cos(2*np.pi*xinterval) # transforming original x coordinates into x circumference positions
        _xr0=_x[0]*r[0]
        _xr1=_x[0]*r[1]
        _xr2=_x[1]*r[2]
        _xr3=_x[1]*r[3]
        _y=np.sin(2*np.pi*xinterval) # transforming original x coordinates into y circumference positions
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
        lineres=np.amax([int(np.abs(xinterval[0]-xinterval[1])*_lineres),10])
        if _yr1> 0 and _yr2>0:
            link=np.sqrt(r[1]**2-np.linspace(_xr1, _xr2, lineres)**2)
            plt.plot(np.linspace(_xr1, _xr2, lineres), link, c=_color,linewidth=linewidth, rasterized=True)
        elif _yr1 <0 and _yr2 <0:
            link=-np.sqrt(r[1]**2-np.linspace(_xr1, _xr2, lineres)**2)
            
            plt.plot(np.linspace(_xr1, _xr2, lineres), link, c=_color,linewidth=linewidth, rasterized=True)
        elif _yr1> 0 and _yr2 < 0:
            _r=r[1]
            if _xr1 <0 or _xr2 <0:
                _r=-_r
            link=np.sqrt(r[1]**2-np.linspace(_xr1, _r, lineres)**2)
            plt.plot(np.linspace(_xr1, _r, lineres), link, c=_color,linewidth=linewidth, rasterized=True)
            link=-np.sqrt(r[1]**2-np.linspace(_r, _xr2, lineres)**2)
            plt.plot(np.linspace(_r, _xr2, lineres), link, c=_color,linewidth=linewidth, rasterized=True)
        
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
    
    
    
    if sample_classes!=None:
        assert len(Z2['ivl'])==len(label_coords), "Internal error, label numbers "+str(len(Z2['ivl'])) +" and "+str(len(label_coords))+" must be equal!" 
        
        j=0
        outerrad=R*1.05+width*len(sample_classes)+space*(len(sample_classes)-1)
        #print(outerrad)
        #sort_index=np.argsort(Z2['icoord'])
        #print(sort_index)
        intervals=[]

        labelnames=[]
        colorlabels_legend={}
        for labelname, colorlist in sample_classes.items():
            ucolors=sorted(list(np.unique(colorlist)))
            type_num=len(ucolors)
            _cmp=plt.get_cmap(colormap_list[j])
            _colorlist=[_cmp(ucolors.index(c)/(type_num-1)) for c in colorlist]
            _colorlist=np.array(_colorlist)[Z2['leaves']]
            if j!=0:
                outerrad=outerrad-width-space
            innerrad=outerrad-width
            #print(outerrad, innerrad,_colorlist[:10])
            
            for i in range(len(label_coords)):
                _xl,_yl,_rotl =label_coords[i-1]
                if i==0:
                    _rotl-=360
                _x,_y,_rot =label_coords[i]
                if i==len(label_coords)-1:
                    _xr,_yr,_rotr =label_coords[0]
                    _rotr+=360
                else:
                    _xr,_yr,_rotr =label_coords[i+1]
                
                start=(2*np.pi/360)*_rotl*0.5+(2*np.pi/360)*_rot*0.5
                theta=(2*np.pi/360)*((_rotr*0.5+_rot*0.5)-(_rotl*0.5+_rot*0.5))
                #print(start, theta,_rotl, _rotr)
                #print(start, theta, innerrad, outerrad,10, _colorlist[i])
                baumkuchen(ax, start, theta, innerrad, outerrad,10, _colorlist[i])
                
            # patches, texts =plt.pie(colorpos, colors=_colorlist,
            #         radius=outerrad,
            #         counterclock=True,
            #         startangle=label_coords[0][2]*0.5)
            # circle=plt.Circle((0,0),innerrad, fc='white')
            # plt.gca().add_patch(circle)
            labelnames.append(labelname)
            colorlabels_legend[labelname]={}
            colorlabels_legend[labelname]["colors"]=_cmp(np.linspace(0, 1, type_num))
            colorlabels_legend[labelname]["labels"]=ucolors
            j+=1
        
        if colorlabels_legend!=None:
            for i, labelname in enumerate(labelnames):
                #print(colorlabels_legend[labelname]["colors"])
                colorlines=[]
                for c in colorlabels_legend[labelname]["colors"]:
                    colorlines.append(Line2D([0], [0], color=c, lw=4))
                leg=plt.legend(colorlines,
                           colorlabels_legend[labelname]["labels"],
                       bbox_to_anchor=(1.0, 1.0-0.3*i),
                       title=labelname)
                plt.gca().add_artist(leg)
            #break
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.bottom.set_visible(False)
    #ax.set_rasterization_zorder(None)
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
    plt.subplots_adjust(left=0.05, right=0.75)
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
    hierarchy.set_link_color_palette([matplotlib.colors.rgb2hex(rgb[:3]) for rgb in cmap])
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