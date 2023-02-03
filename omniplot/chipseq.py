from typing import List,Dict,Optional,Union
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pyBigWig as pwg
import os 
import scipy.stats
from scipy.spatial.distance import squareform
from scipy.spatial import distance
from joblib import Parallel, delayed
import itertools as it
from joblib.externals.loky import get_reusable_executor

sns.set_theme(font="Arial", style={'grid.linestyle': "",'axes.facecolor': 'white'})
class gff_parser():
    
    def __init__(self,gff):
        data={}
        with open(gff) as fin:
            for l in fin:
                if l.startswith("#"):
                    continue
                chrom, source, kind, s, e, _, ori, _, meta=l.split()
                if not chrom in data:
                    data[chrom]={"pos":[], "kind":[],"ori":[],"meta":[]}
                
                meta=meta.split(";")
                tmp={}
                for m in meta:
                    k, v=m.split("=")
                    tmp[k]=v
                data[chrom]["pos"].append([int(s)-1, int(e)])
                data[chrom]["kind"].append(kind)
                data[chrom]["ori"].append(ori)
                data[chrom]["meta"].append(tmp)        
        self.data=data
        
    
    def get_genes(self, chrom: str,
                  start: int, 
                  end: int,
                  gene_type: set=set(["protein_coding"])) -> list:
        
        _data=self.data[chrom]
        genes=[]
        for i, (s,e) in enumerate(_data["pos"]):
            if start<=s and e<=end:
                if _data["kind"][i]=="gene" and _data["meta"][i]["gene_type"] in gene_type:
                    genename=_data["meta"][i]["gene_name"]
                    ori=_data["ori"][i]
                    genes.append([genename,s,e,ori])
            elif start<=s<=end and end<e:
                if _data["kind"][i]=="gene"  and _data["meta"][i]["gene_type"] in gene_type:
                    genename=_data["meta"][i]["gene_name"]
                    ori=_data["ori"][i]
                    genes.append([genename,s,end,ori])
            elif s<start and start<=e<=end  and _data["meta"][i]["gene_type"] in gene_type:
                if _data["kind"][i]=="gene":
                    genename=_data["meta"][i]["gene_name"]
                    ori=_data["ori"][i]
                    genes.append([genename,start,e,ori])
            
                
            if s>end:
                break
        return genes
def plot_bigwig(files: dict, 
                bed: str, 
                gff: str,
                step: int=100):
    """
    Plotting bigwig files in specified genomic regions in the bed file.  
    
    Parameters
    ----------
    files : dict
        A dictionary whose keys are sample names and values are file names 
    bed : str
        A bed file name containing the list of genomic regions
    gff : str
        A gff3 file with a corresponding genome version
    step: int
        A bin size to reduce the bigwig signal resolution.
    show : bool
        Whether or not to show the figure.
    Returns
    -------
    {"ax":axes,"values":mat,"genes":geneset,"positions":pos} : dict
        A dictionary containing matplot axes, signal values, geneset, and positions.
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
    pos=[]
    with open(bed) as fin:
        for l in fin:
            l=l.split()
            chrom, s, e=l[0],l[1],l[2]
            pos.append([chrom,int(s.replace(",","")),int(e.replace(",",""))])
    gff_ob=gff_parser(gff)
    
    geneset=[]
    for chrom, start, end in pos:
        geneset.append(gff_ob.get_genes(chrom, start, end))
    
    for i in range(1000):
        prefix_set=set()
        for f in files:
            h, t=os.path.split(f)
            prefix_set.add(t[:i+1])
        if len(prefix_set) >1:
            prefix=t[:i]
            break
    for i in range(1000):
        suffix_set=set()
        for f in files:
            h, t=os.path.split(f)
            suffix_set.add(t[-(i+1):])
        if len(suffix_set) >1:
            suffix_set=t[-i:]
            break
        
        
    mat={}

    for samplename, f in files.items():
        bw=pwg.open(f)
        #h, t=os.path.split(f)
        #samplename=t.replace(prefix, "").replace(suffix_set, "")
        mat[samplename]=[]
        
        for chrom, s, e in pos:
            val=bw.values(chrom, s, e)
            
            mat[samplename].append(val)
    
    fig, axes=plt.subplots(nrows=len(files)+1,ncols=len(pos),gridspec_kw={
                           'height_ratios': [2]*len(files)+[1]})
    sample_order=sorted(mat.keys())
    for si, sample in enumerate(sample_order):
        vals=mat[sample]
        ymax=0
        
        
        print(ymax)
        for posi, (genes, val) in enumerate(zip(geneset, vals)):
            if len(val)==0:
                continue
            ax=axes[si,posi]
            chrom, s, e=pos[posi]
            val=np.array(val)
            vallen=val.shape[0]
            remains=vallen%step
            _e=e-remains
            x=np.arange(s, _e,step)
            
            print(s, e, _e, vallen, step)
            val=val[:vallen-remains].reshape(-1, step).mean(axis=1)
            _ymax=np.amax(val)
            if _ymax >  ymax:
                ymax =_ymax
            
            ax.fill_between(x, val)
            
            
            
            
            if posi==0:
                ax.set_ylabel(sample)

            ax.set_xticks([],labels=[])
            if si==0:
                ax.set_title(chrom+":"+str(s)+"-"+str(e))

                
        occupied=[[] for i in range(8)]
        for posi in range(len(pos)):
            axes[si,posi].set_ylim(0,ymax*1.01)
            if posi >0:
                axes[si,posi].set_yticks([],labels=[])
    
    for posi in range(len(pos)):
        interval=pos[posi][2]-pos[posi][1]
        for genename, gs, ge,ori in geneset[posi]:
            slot=0
            for oi in range(len(occupied)):
                oc=False
                if len(occupied[oi])==0:
                    occupied[oi].append([gs, ge])
                    oc=False
                    slot=oi
                else:
                    for tmps, tmpe in occupied[oi]:
                        if tmps<=gs<= tmpe or tmps<=ge<=tmpe:
                            oc=True
                    if oc==False:
                        occupied[oi].append([gs, ge])
                        slot=oi
                if oc==False:
                    break
            if gs < pos[posi][1]:
                gs=pos[posi][1]
            if ge > pos[posi][2]:
                ge=pos[posi][2]
            axes[si+1,posi].plot([gs, ge], [slot*1,slot*1], color="gray")
            if ori=="+":
                axes[si+1,posi].plot([ge-interval/32, ge], [slot*1+0.1,slot*1], color="gray")
            elif ori=="-":
                axes[si+1,posi].plot([gs, gs+interval/32], [slot*1,slot*1+0.1], color="gray")
            
            axes[si+1,posi].text((gs+ge)/2, slot*1, genename, ha="center")
        for oi, ol in enumerate(occupied):
            if len(ol)==0:
                maxslot=oi
                break
        axes[si+1,posi].set_ylim(-0.5,maxslot+0.5)
        axes[si+1,posi].ticklabel_format(useOffset=False)
        axes[si+1,posi].set_yticks([],labels=[])
    plt.tight_layout(h_pad=-1,w_pad=-1)
    return {"ax":axes,"values":mat,"genes":geneset,"positions":pos}

def calc_pearson2(_ind, _mat):
    _a, _b=_mat[_ind[0]], _mat[_ind[1]]
    _af=_a>np.quantile(_a, 0.75)
    _bf=_b>np.quantile(_b, 0.75)
    #_ab=
    #filt=_ab>0.02
    filt=_af+_bf
    _a=_a[filt]
    _b=_b[filt]
    p=scipy.stats.pearsonr(_a, _b)[0]
    return p
def calc_pearson(_ind, _mat):
    p=scipy.stats.pearsonr(_mat[_ind[0]], _mat[_ind[1]])[0]
    #p=scipy.stats.spearmanr(_mat[_ind[0]], _mat[_ind[1]])[0]
    #p=distance.cdist([_mat[_ind[0]]], [_mat[_ind[1]]])[0][0]
    return p
def plot_bigwig_correlation(files: dict, 
                            chrom: str="chr1",
                            step: int=1000,
                            palette: str="coolwarm",
                            figsize: list=[10,10],
                            annot: bool=True,
                            clustermap_param: dict={}):
    
    """
    Calculate pearson correlations and draw a heatmap for bigwig files.  
    
    Parameters
    ----------
    files : dict
        A dictionary whose keys are sample names and values are file names 
    chrom : str
        A chromsome to cal
    gff : str
        A gff3 file with a corresponding genome version
    step: int
        A bin size to reduce the bigwig signal resolution.
    show : bool
        Whether or not to show the figure.
    Returns
    -------
    {"ax":axes,"values":mat,"genes":geneset,"positions":pos} : dict
        A dictionary containing matplot axes, signal values, geneset, and positions.
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
    mat=[]
    samples=[]
    bw=pwg.open(list(files.values())[0])
    chrom_sizes=bw.chroms()
    if chrom !="all":
        chrom_size=chrom_sizes[chrom]
        
        for k, f in files.items():
            bw=pwg.open(f)
            val=np.array(bw.values(chrom,0,chrom_size))
            rem=chrom_size%step
            val=val[:val.shape[0]-rem]
            print(val.shape)
            val=np.nan_to_num(val)
            val=val.reshape([-1,step]).mean(axis=1)
            print(val.shape)
            mat.append(val/np.sum(val))
            samples.append(k)
    else:
        chroms=chrom_sizes.keys()
        for k, f in files.items():
            bw=pwg.open(f)
            tmp=[]
            for chrom in chroms:
                chrom_size=chrom_sizes[chrom]
                val=np.array(bw.values(chrom,0,chrom_size))
                rem=chrom_size%step
                val=val[:val.shape[0]-rem]
                val=np.nan_to_num(val)
                val=val.reshape([-1,step]).mean(axis=1)
                tmp.append(val/np.sum(val))
            
            samples.append(k)
            mat.append(np.concatenate(tmp))
            
    mat=np.array(mat)
    dmat=Parallel(n_jobs=-1)(delayed(calc_pearson)(ind, mat) for ind in list(it.combinations(range(mat.shape[0]), 2)))
    dmat=np.array(dmat)
    get_reusable_executor().shutdown(wait=True)
    dmat=squareform(dmat)
    print(dmat)
    dmat+=np.identity(dmat.shape[0])
    sns.clustermap(data=dmat,xticklabels=samples,yticklabels=samples,
               method="ward", cmap=palette,
               col_cluster=True,
               row_cluster=True,
               figsize=figsize,
               rasterized=True,cbar_kws={"label":"Pearson correlation"}, annot=annot,**clustermap_param)
    return {"correlation": dmat,"samples":samples}
    
    
def plot_average():
    pass


if __name__=="__main__":
    #test="plot_bigwig"
    test="plot_bigwig_correlation"
    import glob
    if test=="plot_bigwig":
        fs= {"KMT2A":"/media/koh/grasnas/home/data/omniplot/HepG2_KMT2A-human_ENCFF406SHU.bw",
             "KMT2B":"/media/koh/grasnas/home/data/omniplot/HepG2_KMT2B-human_ENCFF709UTL.bw"}
        bed="/media/koh/grasnas/home/data/omniplot/tmp3.bed"
        gff="/media/koh/grasnas/home/data/omniplot/gencode.v40.annotation.gff3"
        plot_bigwig(fs,bed,gff, step=10)
        plt.show()
    if test=="plot_bigwig_correlation":
        fs= {"KMT2A":"/media/koh/grasnas/home/data/omniplot/HepG2_KMT2A-human_ENCFF406SHU.bw",
             "KMT2B":"/media/koh/grasnas/home/data/omniplot/HepG2_KMT2B-human_ENCFF709UTL.bw",
             "HNF1A_rep1":"/media/koh/grasnas/home/data/omniplot/HepG2_HNF1A-human_ENCFF397BTX.bw",
             "HNF1A_rep2":"/media/koh/grasnas/home/data/omniplot/HepG2_HNF1A-human_ENCFF502ACF.bw",
             "HNF4A_rep1":"/media/koh/grasnas/home/data/omniplot/HepG2_HNF4A-human_ENCFF467SMI.bw",
             "HNF4A_rep2":"/media/koh/grasnas/home/data/omniplot/HepG2_HNF4A-human_ENCFF712VYA.bw",}
        
        plot_bigwig_correlation(fs,step=1000)
        plt.show()

    #raise NotImplementedError("This function will plot ChIP-seq data.")