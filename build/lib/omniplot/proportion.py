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
from omniplot.utils import * #
import scipy.stats as stats
from joblib import Parallel, delayed
from omniplot.chipseq_utils import _calc_pearson
import itertools as it
from omniplot.scatter import *
colormap_list: list=["nipy_spectral", "terrain","tab20b","tab20c","gist_rainbow","hsv","CMRmap","coolwarm","gnuplot","gist_stern","brg","rainbow","jet"]
hatch_list: list = ['//', '\\\\', '||', '--', '++', 'xx', 'oo', 'OO', '..', '**','/o', '\\|', '|*', '-\\', '+o', 'x*', 'o-', 'O|', 'O.', '*-']
maker_list: list=['.', '_' , '+','|', 'x', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'D', 'd', 'P', 'X','o', '1', '2', '3', '4','|', '_']

plt.rcParams['font.family']= 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['svg.fonttype'] = 'none'
sns.set_theme(font="Arial")

__all__=["stacked_barplot","nice_piechart","nice_piechart_num","stackedlines"]

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
    test_pairs : list, optional
        pairs of categorical values related to x. It will calculate -log10 (p value) (mlp) of the fisher exact test.
        Examples: [["Adelie","Chinstrap" ],
                    ["Gentoo","Chinstrap" ],
                    ["Adelie","Gentoo" ]]
    palette : str or dict, optional (default: "tab20c")
        A matplotlib colormap name or dictionary in which keys are values of the hue category and values are RGB array.
        e.g.) palette={""}
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
