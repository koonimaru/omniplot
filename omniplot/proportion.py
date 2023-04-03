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
from matplotlib.patches import Patch
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
from matplotlib.ticker import StrMethodFormatter
from omniplot.scatter import *
colormap_list: list=["nipy_spectral", "terrain","tab20b","tab20c","gist_rainbow","hsv","CMRmap","coolwarm","gnuplot","gist_stern","brg","rainbow","jet"]
hatch_list: list = ['//', '\\\\', '||', '--', '++', 'xx', 'oo', 'OO', '..', '**','/o', '\\|', '|*', '-\\', '+o', 'x*', 'o-', 'O|', 'O.', '*-']
maker_list: list=['.', '_' , '+','|', 'x', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'D', 'd', 'P', 'X','o', '1', '2', '3', '4','|', '_']

plt.rcParams['font.family']= 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['svg.fonttype'] = 'none'
sns.set_theme(font="Arial")

__all__=["stacked_barplot","nice_piechart","nice_piechart_num","stackedlines","nested_piechart","_nested_piechart"]

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


def _float2hist(val, bin_num=10):
    nans=np.isnan(val)
    ranges=[]
    allcounts=[]

    if np.sum(nans)>0:
        val=val[~nans]
        nancount=np.sum(nans)
        ranges.append("NA")
        allcounts.append(nancount)

    n=(np.amax(val)-np.amin(val))/bin_num
    binsize=round(n,-round(np.log10(n)))
    _n=np.abs(np.amin(val))
    minval=round(_n,-round(np.log10(n)))
    if np.amin(val)<0:
        minval-=binsize
    _n=np.abs(np.amax(val))
    maxval=round(_n,-round(np.log10(n)))
    bins=np.linspace(minval, maxval, bin_num+1)
    counts, _=np.histogram(val, bins=bins)
    for i in range(counts.shape[0]):
        if np.abs(bins[i])<0.001 or np.abs(bins[i])>=10**3:
            num1="{:.2E}".format(bins[i])
            num2="{:.2E}".format(bins[i+1])
        else:
            num1=str(bins[i])
            num2=str(bins[i+1])
        ranges.append(num1+"-"+num2)
        allcounts.append(counts[i])
    return ranges, allcounts


def _float2cat(val, bin_num=10):
    nans=np.isnan(val)
    if np.sum(nans)>0:
        _val=val[~nans]
    else:
        _val=val
    n=(np.amax(_val)-np.amin(_val))/bin_num
    binsize=round(n,-round(np.log10(n)))
    _n=np.abs(np.amin(_val))
    minval=round(_n,-round(np.log10(n)))
    if np.amin(_val)<minval:
        

        minval-=binsize
        if np.amin(_val) >0 and minval <0:
            minval=0
    _n=np.abs(np.amax(_val))
    maxval=round(_n,-round(np.log10(n)))
    if maxval<np.amax(_val):
        maxval+=binsize
    bins=[ minval+i*binsize for i in range(bin_num) if minval+i*binsize<=maxval]
    bindict={}
    for b in bins[:-1]:
        if binsize<0.001 or binsize>=10**3:
            num1="{:.2E}".format(b)
            num2="{:.2E}".format(b+binsize)
        else:
            num1=str(b)
            num2=str(b+binsize)
        bindict[b]=num1+"-"+num2
    val_string=[]

    for v in val:
        if np.isnan(v)==True:
            val_string.append("NA")
        else:
            label=""
            for i in range(len(bins)-1):
                if bins[i] <=v< bins[i]+binsize:
                    label=bindict[bins[i]]
                    break
            if label=="":
                if v==maxval:
                    label=bindict[bins[i]]
            val_string.append(label)

    return val_string



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
                    show_legend:bool=True,
                    bin_num: Union[dict, int]=10,
                    ylim: Optional[int]=None)-> Dict:
    
    """
    Drawing a stacked barplot with or without the fisher's exact test 
    
    Parameters
    ----------
    df : pandas DataFrame
    
    x: str or list
        The category to place in x axis. Multiple categories can be passed by a list. 
        It mainly works with categorical values, including strings boolians and integers, but also can take float values by automatically creating a histogram.
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
    title: str optional, (default:"")
        The title of the figure.
    hatch: bool, optional (default: False)
        Adding hatches to the bars
    rotation: int, optional (default:90)
        The orientation of the x axis labels.
    ax: plt.Axes, optional (default: None)
        The ax object to be plotted.
    show_legend: bool, optional (default: True)
        Whether to show legends.
    bin_num: dict, int, optional (default: 10)
        A histogram bin number when columns with float values are selected. 
        You can specify the bin number of each column by using dictionary (e.g., bin_num={"A":10,"B",5}).

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
    if type(df)==Dict:
        df=pd.DataFrame(df)
    if type(x)==str:
        x=[x]
        if order!=None:
            order=[order]
    if type(hue)==str:
        hue=[hue]
        if hue_order!=None:
            hue_order=[hue_order]
    
    for _x in x:

        if df[_x].dtype==float:
            if type(bin_num)==dict:
                df[_x]=_float2cat(df[_x].values,bin_num=bin_num[_x])
            else:
                df[_x]=_float2cat(df[_x].values,bin_num=bin_num)

        elif df[_x].isnull().values.any():
            df[_x]=df[_x].replace(np.nan, "NA")
    for _hue in hue:
        if df[_hue].dtype==float:
            if type(bin_num)==dict:
                df[_hue]=_float2cat(df[_hue].values,bin_num=bin_num[_x])
            else:
                df[_hue]=_float2cat(df[_hue].values,bin_num=bin_num)

        elif df[_hue].isnull().values.any():
            df[_hue]=df[_hue].replace(np.nan, "NA")
    data: dict={}
    xkeys: dict={}
    keysx: dict={}
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
            if len(keys)==1:
                ax.set_xticks(ax.get_xticks(), labels=keys, rotation=rotation)
                ax.margins(x=1)
            else:
                ax.set_xticks(ax.get_xticks(), labels=keys, rotation=rotation)
            #ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            if show_legend==True:
                ax.legend(title=_hue,loc=[1.01,0])
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
            if scale=="absolute" and ylim !=None:
                ax.set_ylim(0, ylim)

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
                  order: str="largest",
                  ncols: int=2,
                  ignore: float=0.05,
                  show_values: bool=True,
                  title: str="",
                  hatch: bool=False,
                  figsize: list=[],
                  show_legend:bool=False,bbox_to_anchor: list=[1.1, 1],
                  right: float=0.7,bottom=0.1) ->Dict:
    """
    Drawing a nice pichart by counting the occurrence of values from pandas dataframe. if you want to draw a pie chart from numerical values, try nice_piechart_num.
    
    Parameters
    ----------
    df : pandas DataFrame

    category: str or list
        The column names for counting values.

    order: str, optional (default: "largest") ["largest", "smallest"]
        How to sort values in the chart
    ncols: int, optional (default: 2)
        The column number of the figure.
    ignore : float, optional (default: 0.05)
        Remove annotations of which the fraction is smaller than this value. 
    palette : str or dict, optional (default: "tab20c")
        A matplotlib colormap name or dictionary in which keys are values of the hue category and values are RGB array.
        e.g.) palette={""}
    show_values: bool, optional
        Wheter to exhibit the values of fractions/counts/percentages.
    
    show : bool, optional
        Whether or not to show the figure.
    
    figsize : List[int], optional
        The figure size, e.g., [4, 6].
    title: str optional, (default:"")
        The title of the figure.
    hatch: bool, optional (default: False)
        Adding hatches to the bars
    show_legend: bool, optional (default: False)
        Whether to show legends.
    bin_num: dict, int, optional (default: 10)
        A histogram bin number when columns with float values are selected. 
        You can specify the bin number of each column by using dictionary (e.g., bin_num={"A":10,"B",5}).

    Returns
    -------
    {"axes":ax}: dict
    
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
    if type(category)==str:
        category=[category]
    nrows=len(category)//ncols+int(len(category)%ncols!=0)
    if show_legend==True or hatch==True:
        if len(figsize)==0:
            figsize=[ncols*4,nrows*3]
        fig, axes=plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    else:
        if len(figsize)==0:
            figsize=[ncols*3,nrows*3]
        fig, axes=plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if title !="":
        fig.suptitle(title)

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
        if hatch==True:
            ax.pie(c, labels=u, 
                counterclock=False,
                startangle=90, 
                colors=colors,
                radius=1.25,
                hatch=hatch_list[:c.shape[0]],
                labeldistance=None)
            ax.legend(bbox_to_anchor=[1,1])
        else:
            if show_legend==True:
                ax.pie(c, labels=u, 
                    counterclock=False,
                    startangle=90, 
                    colors=colors,
                    radius=1.25,
                    labeldistance=None)
                ax.legend(bbox_to_anchor=[1,1])
            else:
                ax.pie(c, labels=u, 
                    counterclock=False,
                    startangle=90, 
                    colors=colors,
                    labeldistance=0.6,
                    radius=1.25)
        ax.set_title(cat,backgroundcolor='lavender',pad=10)
    


    if len(category)!=ncols*nrows:
        for i in range(ncols*nrows-len(category)):
            fig.delaxes(axes[-(i+1)])
    # plt.tight_layout(h_pad=1)
    plt.subplots_adjust(top=0.9,wspace=0.5)
    if title !="":
        fig.suptitle(title)
    return {"axes":ax}


def nice_piechart_num(df: pd.DataFrame,
                      variables: List[str]=[],
                      category: str="" ,
                  
                  palette: str="tab20c",
                  ncols: int=2,
                  ignore: float=0.05,
                  show_values: bool=True,
                  title: str="",
                  figsize=[]) ->Dict:
    """
    Drawing a nice pichart by taking numerical values.
    
    Parameters
    ----------
    df : pandas DataFrame
        It must be a wide-form data. Each pie chart will be drawn from each row, and by default, index values are used for pie chart titles. 
        
    variables: list, optional
        The column names to calculate fractions. If not given, all columns will be used for calculation (in this case, all columns must contain only numerical values).
    
    order: str, optional (default: "largest") ["largest", "smallest"]
        How to sort values in the chart
    ncols: int, optional (default: 2)
        The column number of the figure.
    ignore : float, optional (default: 0.05)
        Remove annotations of which the fraction is smaller than this value. 
    palette : str or dict, optional (default: "tab20c")
        A matplotlib colormap name or dictionary in which keys are values of the hue category and values are RGB array.
        e.g.) palette={""}
    show_values: bool, optional
        Wheter to exhibit the values of fractions/counts/percentages.
    
    show : bool, optional
        Whether or not to show the figure.
    
    figsize : List[int], optional
        The figure size, e.g., [4, 6].
    title: str optional, (default:"")
        The title of the figure.
    hatch: bool, optional (default: False)
        Adding hatches to the bars
    show_legend: bool, optional (default: True)
        Whether to show legends.
 
    Returns
    -------
    dict {"axes":ax}
    
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
    if category=="":
        category=list(df.index)
    else:
        df=df.set_index(category)
        category=list(df.index)
    if len(variables)!=0:
        df=df[variables]
        
    srt=np.argsort(df.sum(axis=0))[::-1]
    df=df[df.columns[srt]]
    variables=list(df.columns)
    nrows=len(category)//ncols+int(len(category)%ncols!=0)
    if len(figsize)==0:
        figsize=[ncols*2,nrows*2]
    fig, axes=plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes=axes.flatten()
    _cmap=plt.get_cmap(palette, len(variables))
    colors=[_cmap(i) for i in range(len(variables))]
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
                      for color, huelabel in zip(colors, variables)]
    
    fig.legend(handles=legend_elements,bbox_to_anchor=(1, 1))
    if title!="":
        plt.title(title)
    return {"axes":ax}

def _nested_piechart(df: pd.DataFrame, 
                  category: List[str],
                  variable: str="",
                  palette: str="tab20b",
                  figsize=[],
                  ignore: float=0.05,
                  show_values: bool=False,unit: str="",
                  show_percentage: bool=False,
                  colormode: str="independent",
                  ax: Optional[plt.Axes]=None,
                  title: str="",
                  hatch: bool=False,
                  show_legend:bool=False,
                  bbox_to_anchor: list=[1.1, 1],
                  right: float=0.7,bottom=0.1,
                  skip_na: bool=False,
                  order: str="smallest",) ->Dict:
    """
    Drawing a nested pichart by counting the occurrence of string (or integer/boolian) values from pandas dataframe.
    
    Parameters
    ----------
    df : pandas DataFrame

    category: str or list
        The column names for counting values. The order of names will correspond to the hierarchical order of the pie chart.

    order: str, optional (default: "largest") ["largest", "smallest"]
        How to sort values in the chart
    
    ignore : float, optional (default: 0.05)
        Remove annotations of which the fraction is smaller than this value. 
    palette : str or dict, optional (default: "tab20c")
        A matplotlib colormap name or dictionary in which keys are values of the hue category and values are RGB array.
        e.g.) palette={""}
    show_values: bool, optional
        Wheter to exhibit the values of fractions/counts/percentages.
    
    show : bool, optional
        Whether or not to show the figure.
    
    figsize : List[int], optional
        The figure size, e.g., [4, 6].
    title: str optional, (default:"")
        The title of the figure.
    hatch: bool, optional (default: False)
        Adding hatches to the bars
    show_legend: bool, optional (default: False)
        Whether to show legends.
   

    Returns
    -------
    {"axes":ax}: dict
    
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
    #df=df.fillna("NA")
    alllabels={}
    alllabel_sets=set()
    for i, cat in enumerate(category):
        alllabels[i]=[]
        for u in df[cat].unique():
            alllabels[i].append(u)
            alllabel_sets.add(u)
    alllabel_sets=sorted(list(alllabel_sets))
    
    if colormode=="independent":
        _cmap=plt.get_cmap(palette, len(alllabel_sets))
        color_lut={u: _cmap(i) for i, u in enumerate(alllabel_sets)}
        marker_lut={u: hatch_list[i] for i, u in enumerate(alllabel_sets)}
    elif colormode=="hierarchical" or colormode=="top":
        _cmap=plt.get_cmap(palette, len(alllabels[0]))
        color_lut={u: _cmap(i) for i, u in enumerate(alllabels[0])}
    


    data={}
    for i in range(len(category)):
        data[i]={"labels":[],"counts":[]}
        for p in it.product(*[alllabels[j] for j in range(i+1)]):
            tmp=df
            for k, _p in enumerate(p):
                
                tmp=tmp.loc[tmp[category[k]]==_p]
            #print(p, len(tmp))
            if variable !="":
                data[i]["counts"].append(tmp[variable].sum())
            else:
                data[i]["counts"].append(len(tmp))
            data[i]["labels"].append(p)
    if ax==None:
        if len(figsize)==0:
            if (show_legend==True or hatch==True) and colormode=="independent":
                figsize=[8,6]
            else:
                figsize=[6,6]
        fig, ax = plt.subplots(figsize =figsize)
    else:
        fig=None
    height=1
    _bottom=0
    for i, d in data.items():
        x=np.array(d["counts"])
        y=np.ones(x.shape)*height
        percent=np.round(100*x/np.sum(x),2)
        xpi=x/np.sum(x)*2*np.pi
        s=np.pi/2
        label=d["labels"]
        #print(label, x)
        for i, (_x,_label) in enumerate(zip(xpi,label)):
            if skip_na==True and _label[-1]=="NA":
                s-=_x
                continue
            if colormode=="independent":
                _color=color_lut[_label[-1]]
                edgecolor="white"
            elif colormode=="top":
                _color=color_lut[_label[0]]
                edgecolor="white"
            elif colormode=="hierarchical":
                _color=np.array(color_lut[_label[0]])
                grad=(1-np.amin(_color))/(len(category)-1)
                _color=_color+grad*(len(_label)-1)/len(category)
                _color=np.where(_color>1,1.0,_color)
                edgecolor="darkgray"
            
            if hatch==True:
                _baumkuchen(ax,s, -_x, _bottom, _bottom+height, int(100*_x/np.sum(xpi)), _color,edgecolor=edgecolor, linewidth =1,hatch=marker_lut[_label[-1]])
            else:
                _baumkuchen(ax,s, -_x, _bottom, _bottom+height, int(100*_x/np.sum(xpi)), _color,edgecolor=edgecolor, linewidth =1)


            txt=""
            if show_legend==False or colormode!="independent":
                txt= _label[-1]
            
            if show_values==True:
                if x[i]>10000 or x[i] < 0.0001:
                    txt=txt+"\n{:.3E} {}".format(x[i], unit)
                else:
                    
                    txt=txt+"\n{:.3f} {}".format(x[i], unit)
            if show_percentage==True:
                txt=txt+"\n({}%)".format(percent[i])
            txt=txt.strip("\n")
            showlabel=False
            if show_legend==False:
                showlabel=True
            elif hatch==False:
                showlabel=True
            elif colormode!="independent":
                showlabel=True
            
            if _x/(2*np.pi)>=ignore and showlabel==True:
                ax.text(np.cos(s-_x/2)*(_bottom+height/2), np.sin(s-_x/2)*(_bottom+height/2),txt, ha="center",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="y", lw=1, alpha=0.8))
            elif _x/(2*np.pi)>=ignore and (show_values==True or show_percentage==True):
                
                ax.text(np.cos(s-_x/2)*(_bottom+height/2), np.sin(s-_x/2)*(_bottom+height/2),txt, ha="center",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="y", lw=1, alpha=0.8))
                

            s-=_x
            
        _bottom+=height
        height=height*0.6
    if show_legend==True or hatch==True:
        if colormode=="independent":
            for i, cat in enumerate(category):
                
                if hatch==True:
                    handles = [
                        Patch(facecolor=color_lut[label], label=label, hatch=marker_lut[label]) 
                        for label in alllabels[i]]
                else:
                    handles = [
                        Patch(facecolor=color_lut[label], label=label) 
                        for label in alllabels[i]]
                legend=ax.legend(handles=handles,bbox_to_anchor=[bbox_to_anchor[0], bbox_to_anchor[0]-i*0.3], title=cat, loc="center left")
                #legend=fig.legend(handles=handles,loc="outside right upper", title=cat)
                #bbox_to_anchor[1]-=0.3
                ax.add_artist(legend)
            plt.subplots_adjust(right=right, bottom=bottom)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    if title!="":
        plt.title(title)
    return ax



def nested_piechart(df: pd.DataFrame, 
                  category: List[str],
                  variable: str="",
                  palette: str="tab20b",
                  figsize: list=[],
                  ignore: float=0.05,
                  show_values: bool=False,unit: str="",
                  show_percentage: bool=False,
                  colormode: str="independent",
                  ax: Optional[plt.Axes]=None,
                  title: str="",
                  hatch: bool=False,
                  show_legend:bool=False,
                  bbox_to_anchor: list=[1.1, 1],
                  right: float=0.7,
                  bottom: float=0.1,
                  skip_na: bool=False,
                  order: str="intact",) ->Dict:
    """
    Drawing a nested pichart by counting the occurrence of string (or integer/boolian) values from pandas dataframe.
    
    Parameters
    ----------
    df : pandas DataFrame

    category: str or list
        The column names for counting values. The order of names will correspond to the hierarchical order of the pie chart.

    order: str, optional (default: "largest") ["largest", "smallest", "intact"]
        How to sort values in the chart
    
    ignore : float, optional (default: 0.05)
        Remove annotations of which the fraction is smaller than this value. 
    
    palette : str or dict, optional (default: "tab20b")
        A matplotlib colormap name or dictionary in which keys are values of the hue category and values are RGB array.
        e.g.) palette={""}
    
    show_values: bool, optional
        Wheter to exhibit values/counts.

    show_percentage: bool, optional
        Wheter to exhibit percentages.
    
    show : bool, optional
        Whether or not to show the figure.
    
    figsize : List[int], optional
        The figure size, e.g., [4, 6].
    title: str optional, (default:"")
        The title of the figure.
    hatch: bool, optional (default: False)
        Adding hatches to the bars
    show_legend: bool, optional (default: False)
        Whether to show legends.
    colormode: str, optional (default: "independent") ["independent", "hierarchical", "top"]
        Choosing the ways to color labels. "independent" will give each label a different color (from a single color palette).
        "top" will give different colors to only labels in the top, and other labels follow the same color with the top category they belong to.
        "hierarchical" is similar to "top", but give fainter colors to labels in the lower hierarchical categories.
    
    bbox_to_anchor: list, optional (default:[1.1, 1])
        the location of legends to appear
    right: float, optional (default: 0.7)
        In the case that legends are truncated by the frame of the figure, decrease this value.

    bottom: float, optional (default: 0.1)

    skip_na: bool, optional (default: False)
        Whether to skip drawing samples labeled "NA". If True, there will be empty spaces for samples with "NA".


    Returns
    -------
    {"axes":ax}: dict
    
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
    #df=df.fillna("NA")
    alllabels={}
    alllabel_sets=set()
    for i, cat in enumerate(category):
        alllabels[i]=[]
        for u in df[cat].unique():
            alllabels[i].append(u)
            alllabel_sets.add(u)
    alllabel_sets=sorted(list(alllabel_sets))
    
    if colormode=="independent":
        _cmap=plt.get_cmap(palette, len(alllabel_sets))
        color_lut={u: _cmap(i) for i, u in enumerate(alllabel_sets)}
        marker_lut={u: hatch_list[i] for i, u in enumerate(alllabel_sets)}
    elif colormode=="hierarchical" or colormode=="top":
        _cmap=plt.get_cmap(palette, len(alllabels[0]))
        color_lut={u: _cmap(i) for i, u in enumerate(alllabels[0])}
    


    # counting data in each level of categories.
    # {0: {'labels': [('Adelie',), ('Chinstrap',), ('Gentoo',)],
    #   'counts': [152, 68, 124]},
    #  1: {'labels': [('Adelie', 'Male'),
    #    ('Adelie', 'Female'),
    #    ('Adelie', 'NA'),
    #    ('Chinstrap', 'Male'),
    #    ('Chinstrap', 'Female'),
    #    ('Chinstrap', 'NA'),
    #    ('Gentoo', 'Male'),
    #    ('Gentoo', 'Female'),
    #    ('Gentoo', 'NA')],
    #   'counts': [73, 73, 6, 34, 34, 0, 61, 58, 5]}}


    data={}
    for i in range(len(category)):
        data[i]={"labels":[],"counts":[]}
        for p in it.product(*[alllabels[j] for j in range(i+1)]):
            tmp=df
            for k, _p in enumerate(p):
                
                tmp=tmp.loc[tmp[category[k]]==_p]
            #print(p, len(tmp))
            if variable !="":
                data[i]["counts"].append(tmp[variable].sum())
            else:
                data[i]["counts"].append(len(tmp))
            data[i]["labels"].append(p)
    if ax==None:
        if len(figsize)==0:
            if (show_legend==True or hatch==True) and colormode=="independent":
                figsize=[8,6]
            else:
                figsize=[6,6]
        fig, ax = plt.subplots(figsize =figsize)
    else:
        fig=None
    
    # Reorganizing data in order to sort the counts
    # 
    _data2=[]
    lastk=np.amax(list(data.keys()))
    #print(lastk)
    for k, v in data.items():
        v["counts"]=np.array(v["counts"])
        v["labels"]=np.array(v["labels"])
        if order=="largest":
            srt=np.argsort(v["counts"])[::-1]
            v["counts"]=v["counts"][srt]
            v["labels"]=v["labels"][srt]
        elif order=="smallest":
            srt=np.argsort(v["counts"])
            v["counts"]=v["counts"][srt]
            v["labels"]=v["labels"][srt]
        elif order=="intact":
            pass
        else:
            raise Exception("Unknown order type.") 

        tmp=[]
        latmp=[]
        cotmp=[]
        if len(_data2)==0:
            for l in v["labels"]:
                _l="|:|".join(l)
                latmp.append(_l)
                tmp.append({_l:{"labels":[],"counts":[]}})
        else:
            # a brutal way to resort data according to the parent category order
            # this will not work for a large dataset
            for parentd in _data2[-1]:                             
                key=list(parentd.keys())[0]
                for l, c in zip(v["labels"], v["counts"]):
                    _l="|:|".join(l)
                    if _l.startswith(key):
                        latmp.append(_l)
                        tmp.append({_l:{"labels":[],"counts":[]}}) 
                        cotmp.append(c)
        if len(_data2)!=0:
            for la, co in zip(latmp, cotmp):

                parent=la.rsplit("|:|",1)[0]
                for parentd in _data2[-1]:
                    if parent in parentd:
                        #parentd[parent].append([la, co])
                        parentd[parent]["labels"].append(la)
                        parentd[parent]["counts"].append(co)
        else:
            parent="top"
            parentd={"top":{"labels":[],"counts":[]}}
            for la, co in zip(latmp, v["counts"]):
                parentd[parent]["labels"].append(la)
                parentd[parent]["counts"].append(co)
            _data2.append([parentd])
        if lastk==k:
            break
        _data2.append(tmp)
    #print(data)
    print(_data2)
    # Drawing the pie chart. 
    # _data2 looks like the bellow. A nested list ordered from the top to bottom hierarchical categories (in the below, (species, sex, island)).
    # The element of the list is also a list consists of dictionaries (if sorted, ordered according to the parent category) of which keys are from
    # the parent category. And the values are dictionaries containing labels and counts. These labels are combination of those in the parent and 
    # present categories. 
    #  
    #  
    # [
    # [{'top': {'labels': ['Adelie', 'Gentoo', 'Chinstrap'], 'counts': [152, 124, 68]}}], 
    # 
    # [{'Adelie': {'labels': ['Adelie|:|Female', 'Adelie|:|Male', 'Adelie|:|NA'], 'counts': [73, 73, 6]}}, 
    #  {'Gentoo': {'labels': ['Gentoo|:|Male', 'Gentoo|:|Female', 'Gentoo|:|NA'], 'counts': [61, 58, 5]}}, 
    #  {'Chinstrap': {'labels': ['Chinstrap|:|Female', 'Chinstrap|:|Male', 'Chinstrap|:|NA'], 'counts': [34, 34, 0]}}], 
    # 
    # [{'Adelie|:|Female': {'labels': ['Adelie|:|Female|:|Dream', 'Adelie|:|Female|:|Torgersen', 'Adelie|:|Female|:|Biscoe'], 'counts': [27, 24, 22]}}, 
    #  {'Adelie|:|Male': {'labels': ['Adelie|:|Male|:|Dream', 'Adelie|:|Male|:|Torgersen', 'Adelie|:|Male|:|Biscoe'], 'counts': [28, 23, 22]}}, 
    #  {'Adelie|:|NA': {'labels': ['Adelie|:|NA|:|Torgersen', 'Adelie|:|NA|:|Dream', 'Adelie|:|NA|:|Biscoe'], 'counts': [5, 1, 0]}}, 
    #  {'Gentoo|:|Male': {'labels': ['Gentoo|:|Male|:|Biscoe', 'Gentoo|:|Male|:|Torgersen', 'Gentoo|:|Male|:|Dream'], 'counts': [61, 0, 0]}}, 
    #  {'Gentoo|:|Female': {'labels': ['Gentoo|:|Female|:|Biscoe', 'Gentoo|:|Female|:|Torgersen', 'Gentoo|:|Female|:|Dream'], 'counts': [58, 0, 0]}}, 
    #  {'Gentoo|:|NA': {'labels': ['Gentoo|:|NA|:|Biscoe', 'Gentoo|:|NA|:|Dream', 'Gentoo|:|NA|:|Torgersen'], 'counts': [5, 0, 0]}}, 
    #  {'Chinstrap|:|Female': {'labels': ['Chinstrap|:|Female|:|Dream', 'Chinstrap|:|Female|:|Torgersen', 'Chinstrap|:|Female|:|Biscoe'], 'counts': [34, 0, 0]}}, 
    #  {'Chinstrap|:|Male': {'labels': ['Chinstrap|:|Male|:|Dream', 'Chinstrap|:|Male|:|Biscoe', 'Chinstrap|:|Male|:|Torgersen'], 'counts': [34, 0, 0]}}, 
    #  {'Chinstrap|:|NA': {'labels': ['Chinstrap|:|NA|:|Torgersen', 'Chinstrap|:|NA|:|Biscoe', 'Chinstrap|:|NA|:|Dream'], 'counts': [0, 0, 0]}}]
    # ]
    #
    #
    height=1
    _bottom=0
    for h, ds in enumerate(_data2):
        s=np.pi/2

        for h2, dg in enumerate(ds):
            for h3, d in dg.items():
                #print(h, h2,h3,d)
                x=np.array(d["counts"])
                if np.sum(x)==0:
                    continue
                if h==0:
                    total=np.sum(x)
                y=np.ones(x.shape)*height
                percent=np.round(100*x/total,2)
                xpi=x/total*2*np.pi
                
                label=d["labels"]
                
                #print(label, x)
                for i, (_x,_label) in enumerate(zip(xpi,label)):
                    _label=_label.split("|:|")
                    if skip_na==True and _label[-1]=="NA":
                        s-=_x
                        continue
                    if colormode=="independent":
                        _color=color_lut[_label[-1]]
                        edgecolor="white"
                    elif colormode=="top":
                        _color=color_lut[_label[0]]
                        edgecolor="white"
                    elif colormode=="hierarchical":
                        _color=np.array(color_lut[_label[0]])
                        grad=(1-np.amin(_color))/(len(category)-1)
                        _color=_color+grad*(len(_label)-1)/len(category)
                        _color=np.where(_color>1,1.0,_color)
                        edgecolor="darkgray"
                    
                    if hatch==True:
                        _baumkuchen(ax,s, -_x, _bottom, _bottom+height, int(100*_x/np.sum(xpi)), _color,edgecolor=edgecolor, linewidth =1,hatch=marker_lut[_label[-1]])
                    else:
                        _baumkuchen(ax,s, -_x, _bottom, _bottom+height, int(100*_x/np.sum(xpi)), _color,edgecolor=edgecolor, linewidth =1)


                    txt=""
                    if show_legend==False or colormode!="independent":
                        txt= _label[-1]
                    
                    if show_values==True:
                        if x[i]>10000 or x[i] < 0.0001:
                            txt=txt+"\n{:.3E} {}".format(x[i], unit)
                        else:
                            
                            txt=txt+"\n{:.3f} {}".format(x[i], unit)
                    if show_percentage==True:
                        txt=txt+"\n({}%)".format(percent[i])
                    txt=txt.strip("\n")
                    showlabel=False
                    if show_legend==False:
                        showlabel=True
                    elif hatch==False:
                        showlabel=True
                    elif colormode!="independent":
                        showlabel=True
                    
                    if _x/(2*np.pi)>=ignore and showlabel:
                        ax.text(np.cos(s-_x/2)*(_bottom+height/2), np.sin(s-_x/2)*(_bottom+height/2),txt, ha="center",
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="y", lw=1, alpha=0.8))
                    elif _x/(2*np.pi)>=ignore and (show_values==True or show_percentage==True):
                        
                        ax.text(np.cos(s-_x/2)*(_bottom+height/2), np.sin(s-_x/2)*(_bottom+height/2),txt, ha="center",
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="y", lw=1, alpha=0.8))
                        

                    s-=_x
            
        _bottom+=height
        height=height*0.6
    

    # Drawing legend. if hatch is True, because it will be too messy to add texts to the pie chart, it will automatically draw legends.

    if show_legend==True or hatch==True:
        if colormode=="independent":
            for i, cat in enumerate(category):
                
                if hatch==True:
                    handles = []
                    for label in alllabels[i]:
                        if skip_na==True and label=="NA":
                            continue
                        handles.append(Patch(facecolor=color_lut[label], label=label, hatch=marker_lut[label]))
                else:
                    handles = []
                    for label in alllabels[i]:
                        if skip_na==True and label=="NA":
                            continue
                        handles.append(Patch(facecolor=color_lut[label], label=label) ) 
                legend=ax.legend(handles=handles,bbox_to_anchor=[bbox_to_anchor[0], bbox_to_anchor[0]-i*0.3], title=cat, loc="center left")
                #legend=fig.legend(handles=handles,loc="outside right upper", title=cat)
                #bbox_to_anchor[1]-=0.3
                ax.add_artist(legend)
            plt.subplots_adjust(right=right, bottom=bottom)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    if title!="":
        plt.title(title)
    return ax



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
                xformat: str="",
                yformat: str="",
                logscalex: bool=False,
                logscaley: bool=False,
                title: str="",
                hatch: bool=False):
    """
    Drawing a scatter plot of which points are represented by pie charts. 
    
    Parameters
    ----------
    df : pandas DataFrame
        A wide form dataframe. It accepts negative values.
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
    alpha: float, optional (default: 0.75)
        alpha of the stacked lines (areas).
    bbox_to_anchor: list, optional (default: [1.7, 1])
        The legend location
    right: float=0.7,
    bottom: float=0.120,
    show_legend: bool=True,
    
    yunit: str="", optional (default: "")
        The y axis unit. it will appear at the top of the y axis.
    xunit: str="", optional (default: "")
        The X axis unit. it will appear at the top of the x axis.
    title: str="", optional (default: "")
        The figure title.
    hatch: bool, optional (default: False)
        Whether to add hatches to the plots.

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
    if xformat!="":
        
        ax.xaxis.set_major_formatter(StrMethodFormatter(xformat))
    if yformat !="":
        ax.yaxis.set_major_formatter(StrMethodFormatter(yformat))
    if logscalex==True:
        ax.set_xscale("log")
    if logscaley==True:
        ax.set_yscale("log")
    if inverse==True:
        ax.invert_yaxis()
    return {"axes":ax}