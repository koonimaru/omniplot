
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
import scipy.stats as stats
colormap_list=["nipy_spectral", "terrain","tab20b","gist_rainbow","tab20c","CMRmap","coolwarm","gnuplot","gist_stern","brg","rainbow"]
plt.rcParams['font.family']= 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['svg.fonttype'] = 'none'
sns.set_theme()
from matplotlib.text import Annotation
from matplotlib.transforms import Affine2D
import os 

__all__=["_create_color_markerlut", 
         "_separate_data", "_line_annotate", "_dendrogram_threshold", "_radialtree2",
         "_get_cluster_classes","_calc_curveture", "_draw_ci_pi","_calc_r2",
         "_ci_pi", "_save","_baumkuchen", "_baumkuchen_xy", "_get_embedding"]


class LineAnnotation(Annotation):
    """A sloped annotation to *line* at position *x* with *text*
    Optionally an arrow pointing from the text to the graph at *x* can be drawn.
    Usage
    -----
    fig, ax = subplots()
    x = linspace(0, 2*pi)
    line, = ax.plot(x, sin(x))
    ax.add_artist(LineAnnotation("text", line, 1.5))
    """

    def __init__(
        self, text, line, x, xytext=(0, 5), textcoords="offset points", **kwargs
    ):
        """Annotate the point at *x* of the graph *line* with text *text*.

        By default, the text is displayed with the same rotation as the slope of the
        graph at a relative position *xytext* above it (perpendicularly above).

        An arrow pointing from the text to the annotated point *xy* can
        be added by defining *arrowprops*.

        Parameters
        ----------
        text : str
            The text of the annotation.
        line : Line2D
            Matplotlib line object to annotate
        x : float
            The point *x* to annotate. y is calculated from the points on the line.
        xytext : (float, float), default: (0, 5)
            The position *(x, y)* relative to the point *x* on the *line* to place the
            text at. The coordinate system is determined by *textcoords*.
        **kwargs
            Additional keyword arguments are passed on to `Annotation`.

        See also
        --------
        `Annotation`
        `line_annotate`
        """
        assert textcoords.startswith(
            "offset "
        ), "*textcoords* must be 'offset points' or 'offset pixels'"

        self.line = line
        self.xytext = xytext

        # Determine points of line immediately to the left and right of x
        xs, ys = line.get_data()

        def neighbours(x, xs, ys, try_invert=True):
            inds, = np.where((xs <= x)[:-1] & (xs > x)[1:])
            if len(inds) == 0:
                assert try_invert, "line must cross x"
                return neighbours(x, xs[::-1], ys[::-1], try_invert=False)

            i = inds[0]
            return np.asarray([(xs[i], ys[i]), (xs[i+1], ys[i+1])])
        
        self.neighbours = n1, n2 = neighbours(x, xs, ys)
        
        # Calculate y by interpolating neighbouring points
        y = n1[1] + ((x - n1[0]) * (n2[1] - n1[1]) / (n2[0] - n1[0]))

        kwargs = {
            "horizontalalignment": "center",
            "rotation_mode": "anchor",
            **kwargs,
        }
        super().__init__(text, (x, y), xytext=xytext, textcoords=textcoords, **kwargs)

    def get_rotation(self):
        """Determines angle of the slope of the neighbours in display coordinate system
        """
        transData = self.line.get_transform()
        dx, dy = np.diff(transData.transform(self.neighbours), axis=0).squeeze()
        return np.rad2deg(np.arctan2(dy, dx))

    def update_positions(self, renderer):
        """Updates relative position of annotation text
        Note
        ----
        Called during annotation `draw` call
        """
        xytext = Affine2D().rotate_deg(self.get_rotation()).transform(self.xytext)
        self.set_position(xytext)
        super().update_positions(renderer)


def _line_annotate(text, line, x, *args, **kwargs):
    """Add a sloped annotation to *line* at position *x* with *text*

    Optionally an arrow pointing from the text to the graph at *x* can be drawn.

    Usage
    -----
    x = linspace(0, 2*pi)
    line, = ax.plot(x, sin(x))
    line_annotate("sin(x)", line, 1.5)

    See also
    --------
    `LineAnnotation`
    `plt.annotate`
    """
    ax = line.axes
    a = LineAnnotation(text, line, x, *args, **kwargs)
    if "clip_on" in kwargs:
        a.set_clip_path(ax.patch)
    ax.add_artist(a)
    return a

def _baumkuchen(ax, start, theta, rin, rout,res, _color,edgecolor="", linewidth=2,hatch=None):
    move=np.linspace(0, theta,res)
    xf=np.concatenate([rin*np.cos(start+move),[rin*np.cos(start+theta),rout*np.cos(start+theta)],rout*np.cos(start+move)[::-1],[rin*np.cos(start),rout*np.cos(start)][::-1]])
    yf=np.concatenate([rin*np.sin(start+move),[rin*np.sin(start+theta),rout*np.sin(start+theta)],rout*np.sin(start+move)[::-1],[rin*np.sin(start),rout*np.sin(start)][::-1]])
    if edgecolor!="":
        ax.plot(xf, yf, zorder=2, color=edgecolor)
    ax.fill(xf, yf, color=_color, linewidth=linewidth,hatch=hatch)

def _baumkuchen_xy(ax, x,y, start, theta, rin, rout,res, _color, edgecolor=""):
    move=np.linspace(0, theta,res)
    xf=np.concatenate([rin*np.cos(start+move),[rin*np.cos(start+theta),rout*np.cos(start+theta)],rout*np.cos(start+move)[::-1],[rin*np.cos(start),rout*np.cos(start)][::-1]])
    yf=np.concatenate([rin*np.sin(start+move),[rin*np.sin(start+theta),rout*np.sin(start+theta)],rout*np.sin(start+move)[::-1],[rin*np.sin(start),rout*np.sin(start)][::-1]])
    if edgecolor!="":
        ax.plot(xf+x, yf+y, zorder=2, color=edgecolor)
    ax.fill(xf+x, yf+y, color=_color, zorder=2)

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
         colorlabels_legend: Optional[dict]=None,
         xticks=set(),ax: Optional[plt.Axes]=None,
         linewidth: float=1,
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
    xticks=set(xticks)
    if figsize==None and colorlabels != None:
        figsize=[7,6]
    elif figsize==None and sample_classes != None:
        figsize=[7,6]
    elif figsize==None :
        figsize=[5,5]
    linewidth=linewidth
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
    xmin = np.amin(Z2["icoord"])
    ymax=np.amax(Z2['dcoord'])
    
    ucolors=sorted(set(Z2["color_list"]))
    #cmap = cm.gist_rainbow(np.linspace(0, 1, len(ucolors)))
    cmp=plt.get_cmap(palette, len(ucolors))
    #print(cmp)
    if type(cmp) == matplotlib.colors.LinearSegmentedColormap:
        cmap = cmp(np.linspace(0, 1, len(ucolors)))
    else:
        cmap=cmp.colors
    if ax ==None:
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
        ax.plot([_xr0, _xr1], [_yr0, _yr1], c=_color,linewidth=linewidth, rasterized=True)
        ax.plot([_xr2, _xr3], [_yr2,_yr3], c=_color,linewidth=linewidth, rasterized=True)
        
        #plotting circular links between nodes
        lineres=np.amax([int(np.abs(xinterval[0]-xinterval[1])*_lineres),10])
        if _yr1> 0 and _yr2>0:
            link=np.sqrt(r[1]**2-np.linspace(_xr1, _xr2, lineres)**2)
            ax.plot(np.linspace(_xr1, _xr2, lineres), link, c=_color,linewidth=linewidth, rasterized=True)
        elif _yr1 <0 and _yr2 <0:
            link=-np.sqrt(r[1]**2-np.linspace(_xr1, _xr2, lineres)**2)
            
            ax.plot(np.linspace(_xr1, _xr2, lineres), link, c=_color,linewidth=linewidth, rasterized=True)
        elif _yr1> 0 and _yr2 < 0:
            _r=r[1]
            if _xr1 <0 or _xr2 <0:
                _r=-_r
            link=np.sqrt(r[1]**2-np.linspace(_xr1, _r, lineres)**2)
            ax.plot(np.linspace(_xr1, _r, lineres), link, c=_color,linewidth=linewidth, rasterized=True)
            link=-np.sqrt(r[1]**2-np.linspace(_r, _xr2, lineres)**2)
            ax.plot(np.linspace(_r, _xr2, lineres), link, c=_color,linewidth=linewidth, rasterized=True)
        
        #Calculating the x, y coordinates and rotation angles of labels

        # _append=False
        # if len(xticks)==0:
        #     _append=y[0]==0
        # else:
        #     _append=x[0] in xticks and y[0]==0

        # if _append==True:
        #     label_coords.append([(1.05+offset)*_xr0, (1.05+offset)*_yr0,360*x[0]/xmax])
        #     #plt.text(1.05*_xr0, 1.05*_yr0, Z2['ivl'][i],{'va': 'center'},rotation_mode='anchor', rotation=360*x[0]/xmax)
        #     i+=1

        # _append=False
        # if len(xticks)==0:
        #     _append=y[3]==0
        # else:
        #     _append=x[3] in xticks and y[3]==0

        # if _append==True:
        # #if y[3]==0 and x[2] in xticks:
        #     label_coords.append([(1.05+offset)*_xr3, (1.05+offset)*_yr3,360*x[2]/xmax])
        #     #plt.text(1.05*_xr3, 1.05*_yr3, Z2['ivl'][i],{'va': 'center'},rotation_mode='anchor', rotation=360*x[2]/xmax)
        #     i+=1
    label_coords = []
    # determine the coordiante of the labels and their rotation:
    for i, label in enumerate(Z2["ivl"]):
        # scipy (1.x.x) places the leaves in x = 5+i*10 , and we can use this
        # to calulate where to put the labels
        place = (5.0 + i * 10.0) / xmax * 2
        label_coords.append(
            [
                np.cos(place * np.pi) * (1.05 + offset),  # _x
                np.sin(place * np.pi) * (1.05 + offset),  # _y
                place * 180,  # _rot
            ]
        )    

    if addlabels==True:
        assert len(Z2['ivl'])==len(label_coords), "Internal error, label numbers "+str(len(Z2['ivl'])) +" and "+str(len(label_coords))+" must be equal!" 
        
        #Adding labels
        for (_x, _y,_rot), label in zip(label_coords, Z2['ivl']):
            ax.text(_x, 
                    _y, 
                    label,
                    {'va': 'center'},
                    rotation_mode='anchor', 
                    rotation=_rot,
                    fontsize=fontsize)
    
    
    
    if sample_classes!=None:
        assert len(Z2['ivl'])==len(label_coords), \
            "Internal error, label numbers "+str(len(Z2['ivl'])) +\
                " and "+str(len(label_coords))+" must be equal!" 
        
        classcounter=0
        outerrad=R*1.05+width*len(sample_classes)+space*(len(sample_classes)-1)
        intervals=[]

        labelnames=[]
        colorlabels_legend={}
        for labelname, colorlist in sample_classes.items():
            ucolors=sorted(list(np.unique(colorlist)))
            type_num=len(ucolors)
            _cmp=plt.get_cmap(colormap_list[classcounter], len(colorlist))
            _colorlist=[_cmp(ucolors.index(c)/(type_num-1)) for c in colorlist]
            #_colorlist=_cmp.colors
            print(len(_colorlist), len(colorlist))
            _colorlist=np.array(_colorlist)[Z2['leaves']]
            if classcounter!=0:
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
                _baumkuchen(ax, start, theta, innerrad, outerrad,10, _colorlist[i] ) #,edgecolor = "gray",linewidth=0.5)
                
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
            classcounter+=1
        
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
    ax.set_xticks([])
    ax.set_yticks([])
    
    if colorlabels!=None:
        maxr=R*1.1+width*len(colorlabels)+space*(len(colorlabels)-1)
    elif sample_classes !=None:
        maxr=R*1.1+width*len(sample_classes)+space*(len(sample_classes)-1)
    else:
        maxr=R*1.1
    ax.set_xlim(-maxr,maxr)
    ax.set_ylim(-maxr,maxr)
    plt.subplots_adjust(left=0.05, right=0.75)
    if show==True:
        plt.show()
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

def _calc_r2(X,Y):
    x_mean = np.mean(X)
    y_mean = np.mean(Y)
    numerator = np.sum((X - x_mean)*(Y - y_mean))
    denominator = ( np.sum((X - x_mean)**2) * np.sum((Y - y_mean)**2) )**.5
    correlation_coef = numerator / denominator
    r2 = correlation_coef**2
    return r2

def _ci_pi(X: np.ndarray,Y: np.ndarray, plotline_X: np.ndarray, y_model: np.ndarray) -> list:
    x_mean = np.mean(X)
    y_mean = np.mean(Y)
    n = X.shape[0]                        # number of samples
    m = 2                             # number of parameters
    dof = n - m                       # degrees of freedom
    t = stats.t.ppf(0.975, dof)       # Students statistic of interval confidence
    residual = Y - y_model
        
    std_error = (np.sum(residual**2) / dof)**.5   # Standard deviation of the error
    # to plot the adjusted model
    x_line = plotline_X.flatten()
    y_line = y_model
    
    # confidence interval
    ci = t * std_error * (1/n + (x_line - x_mean)**2 / np.sum((X - x_mean)**2))**.5
    # predicting interval
    pi = t * std_error * (1 + 1/n + (x_line - x_mean)**2 / np.sum((X - x_mean)**2))**.5
    return ci, pi,std_error
def _draw_ci_pi(ax: plt.Axes, 
               ci: np.ndarray, 
               pi: np.ndarray,
               x_line: np.ndarray, 
               y_line: np.ndarray):
    """
    Drawing a confidence interval and a prediction interval 
    """
    
    
    ax.fill_between(x_line, y_line + pi, y_line - pi, 
                color = 'lightcyan', 
                label = '95% prediction interval',
                alpha=0.5)
    ax.fill_between(x_line, y_line + ci, 
                    y_line - ci, color = 'skyblue', 
                    label = '95% confidence interval',
                    alpha=0.5)
    
from sklearn.cluster import KMeans
def _optimal_kmeans(X: Union[np.ndarray, list], testrange: list, topn: int=2)-> List[int]:
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
    plt.legend()
    print("Top two optimal cluster No are: {}, {}".format(K[srtindex[0]],K[srtindex[1]]))
    n_clusters=[K[srtindex[i]] for i in range(topn)]
    return n_clusters

from matplotlib.backends.backend_pdf import PdfPages

def _multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
    
def _save(save, suffix, fig=None):
    if save !="":
        if save.endswith(".pdf") or save.endswith(".png") or save.endswith(".svg"):
            h, t=os.path.splitext(save)
            if fig!=None:
                fig.savefig(h+"_"+suffix+t)
            else:
                plt.savefig(h+"_"+suffix+t)
        else:
            if fig!=None:
                fig.savefig(h+"_"+suffix+t)
            else:
                
                plt.savefig(save+"_"+suffix+".pdf")


from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.random_projection import SparseRandomProjection

def _get_embedding(method="umap",param={}):
    
    
    defaul_params={"tsne": dict(n_components=2,n_iter=500,
                                n_iter_without_progress=150,
                                perplexity=10,
                                n_jobs=2,
                                random_state=42),
                "random_projection": dict(n_components=2, random_state=42, eigen_solver="arpack"),
                "svd": dict(n_components=2),
                "random_trees": dict(n_estimators=200, max_depth=5, random_state=42),
                "mds": dict(n_components=2, n_init=1, max_iter=120, n_jobs=2, normalized_stress="auto"),
                "lle": dict(n_neighbors=5, n_components=2, ),
                "isomap":dict(n_neighbors=4, n_components=2),
                "linear_discriminant":dict(n_components=2),
                "random_projection": dict(n_components=2, random_state=42),
                "nca": dict(n_components=2, init="pca", random_state=42),
                "umap": dict(min_dist=0.25,n_neighbors=15)}
    if len(param)!=0:
        params={method: param}
    else:
        params=defaul_params
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import RandomTreesEmbedding
    from sklearn.manifold import (
        Isomap,
        LocallyLinearEmbedding,
        MDS,
        SpectralEmbedding,
        TSNE,)
    from sklearn.neighbors import NeighborhoodComponentsAnalysis
    
    if method=="random_projection": 
        embedding=SparseRandomProjection(**params[method]
        )
    elif method=="linear_discriminant": 
        embedding=LinearDiscriminantAnalysis(**params[method])
    elif method=="isomap": 
        embedding=Isomap(**params[method])
    
    elif method=="lle": 
        embedding=LocallyLinearEmbedding(method="standard", **params[method]
        )
    elif method=="modlle": 
        embedding=LocallyLinearEmbedding(method="modified",**params["lle"] 
        )
    elif method=="hessian_lle": 
        embedding=LocallyLinearEmbedding(method="hessian", **params["lle"] 
        )
    elif method=="ltsa_lle": 
        embedding=LocallyLinearEmbedding( method="ltsa", **params["lle"] 
        )
    elif method=="mds": 
        embedding=MDS(**params[method]
        )
    elif method=="random_trees": 
        embedding=make_pipeline(
            RandomTreesEmbedding(**params[method]),
            TruncatedSVD(**params["svd"], ),
        )
    elif method=="spectral": 
        embedding=SpectralEmbedding(**params[method],
        )
    elif method=="tsne": 
        embedding=TSNE(**params[method], 
        )
    elif method=="nca": 
        embedding=NeighborhoodComponentsAnalysis(
            **params[method]
        )
    elif method=="umap":
        import umap 
        embedding=umap.UMAP(**params[method])
    else:
        raise Exception(f"Medthod {method} does not exist.")
    return embedding


def _separate_data(df, variables=[], 
                   category=""):
    if len(variables) !=0:
        x = np.array(df[variables].values)
        if len(category) !=0:
            if type(category)==str:
                category=[category]
            #category_val=df[category].values
        
        #if x.dtype!=np.number: 
        if np.issubdtype(x.dtype, np.number)==False:
            raise TypeError(f"variables must contain only float values. {x.dtype} was given")
    elif len(category) !=0:
        if type(category)==str:
            category=[category]
        #category_val=df[category].values
        df=df.drop(category, axis=1)
        x = np.array(df.values)
        if x.dtype!=np.number: 
            raise TypeError(f"data must contain only float values except {category} column. \
        or you can specify the numeric variables with the option 'variables'.")
        
    else:    
        x = df.values
        #category_val=[]
        if x.dtype!=float: 
            raise TypeError(f"data must contain only float values. \
        or you can specify the numeric variables with the option 'variables'.")
        
    return x, category

def _create_color_markerlut(df, cat, palette, markers=[]):
    color_lut={}
    marker_lut={}
    if df[cat].isnull().values.any():
        df[cat]=df[cat].fillna("NA")
    uniq_labels=sorted(list(set(df[cat])))
    _cmap=plt.get_cmap(palette, len(uniq_labels))
    
    color_lut={u: _cmap(i) for i, u in enumerate(uniq_labels)}
    
    if len(markers)!=0:
        if len(markers) < len(uniq_labels):
            while len(markers) < len(uniq_labels):
                markers.extend(markers)
        marker_lut={u: markers[i] for i, u in enumerate(uniq_labels)}

    return color_lut, marker_lut

