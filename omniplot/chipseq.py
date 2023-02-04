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
import itertools

def range_diff(r1, r2):
    s1, e1 = r1
    s2, e2 = r2
    endpoints = sorted((s1, s2, e1, e2))
    result = []
    if endpoints[0] == s1 and endpoints[1] != s1:
        result.append((endpoints[0], endpoints[1]))
    if endpoints[3] == e1 and endpoints[2] != e1:
        result.append((endpoints[2], endpoints[3]))
    return result

def multirange_diff(r1_list, r2_list):
    for r2 in r2_list:
        r1_list = list(itertools.chain(*[range_diff(r1, r2) for r1 in r1_list]))
    return r1_list

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

def read_bw(f, chrom, chrom_size, step):
        bw=pwg.open(f)
        val=np.array(bw.values(chrom,0,chrom_size))
        rem=chrom_size%step
        val=val[:val.shape[0]-rem]
        val=np.nan_to_num(val)
        val=val.reshape([-1,step]).mean(axis=1)
        return val
def read_peaks(peakfile):
    peaks={}
    with open(peakfile) as fin:
        for l in fin:
            l=l.split()
            if not l[0] in peaks:
                peaks[l[0]]=[]
            peaks[l[0]].append([int(l[1]),int(l[2])])
    return peaks

def stitching(peakfile, stitchdist):
    speaks={}
    
    for chrom, intervals in peakfile.items():
        intervals.sort()
        stack = []
        # insert first interval into stack
        stack.append(intervals[0])
        for i in intervals[1:]:
            # Check for overlapping interval,
            # if interval overlap
            if i[0] - stack[-1][-1] <stitchdist:
                stack[-1][-1] = max(stack[-1][-1], i[-1])
            else:
                stack.append(i)
        speaks[chrom]=stack
    return speaks

def read_tss(tss, tss_dist):
    tss_pos={}
    with open(tss) as fin:
        for l in fin:
            l=l.split()
            if not l[0] in tss_pos:
                tss_pos[l[0]]=[]
            if l[4]=="+":
                tss_pos[l[0]].append([int(l[1])-tss_dist,int(l[1])+tss_dist])
            elif l[4]=="-":
                tss_pos[l[0]].append([int(l[2])-tss_dist,int(l[2])+tss_dist])
            
            else:
                raise Exception("TSS bed file format must look like 'chromsome\tstart\tend\tname\torientation'")
            
    return tss_pos
def remove_close_to_tss(stitched, tss_pos):
    _stitched={}
    chroms=stitched.keys()
    tmp=Parallel(n_jobs=-1)(delayed(multirange_diff)(stitched[chrom], tss_pos[chrom]) for chrom in chroms)
    _stitched={chrom: _tmp for chrom, _tmp in zip(chroms,tmp) }
    # for chrom, se in stitched.items():
    #
    #     tsslist=tss_pos[chrom]
    #     _stitched[chrom]=multirange_diff(se, tsslist)
        
    #
    # for s, e in se:
    #     news=[s]
    #     newe=[e]
    #     tss_inbetween=False
    #     for _tss, ori in tsslist:
    #         if s+tss_dist<_tss:
    #             continue
    #         if _tss >e+tss_dist:
    #             break
    #         if s <= _tss <= e:
    #             newe.append(_tss-tss_dist)
    #             news.append(_tss+tss_dist)
    #             tss_inbetween=True
    #             continue
    #         if 0< _tss-e< tss_dist:
    #             _e=_tss-tss_dist
    #             newe.append(_e)
    #         if 0 < s-_tss < tss_dist:
    #             _s=_tss+tss_dist
    #             news.append(_s)
    #     _e=min(newe)
    #     _s=max(news)
    #     if _s==107683009:
    #         print(chrom, s,e)
    #     if _e <=s:
    #         continue
    #     if e <=_s:
    #         continue
    #     if tss_inbetween:
    #         _stitched[chrom].append([_s,e])
    #         _stitched[chrom].append([s,_e])
    #     else:
    #         _stitched[chrom].append([_s,_e])
    return _stitched

def find_extremes(signals, pos):
    signals=np.array(signals)
    srt=np.argsort(signals)
    signals=signals[srt]
    pos=np.array(pos)
    pos=pos[srt]
    
    y=signals
    _y=y/np.amax(y)
    x=np.linspace(0,1, y.shape[0])
    b=0
    possitives=[]
    for i in reversed(range(x.shape[0])):
        __y=x-i/x.shape[0] + _y[i]
        #print(y)
        possitives.append(np.sum(_y-__y>0))

    index=len(possitives)-np.argmax(possitives)
    
    
    return x, y, pos, index
def plot_bigwig_correlation(files: dict, 
                            chrom: str="chr1",
                            step: int=1000,
                            palette: str="coolwarm",
                            figsize: list=[10,10],
                            annot: bool=True,
                            clustermap_param: dict={},
                            peakfile: str=""):
    
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
    
    
        
    
    if peakfile!="":
        peaks=read_peaks(peakfile)
        for k, f in files.items():
            bw=pwg.open(f)
            tmp=[]
            for chrom, se in peaks.items():
                for s, e in se:
                    val=bw.stats(chrom,s,e, exact=True)[0]
                    if val==None:
                        val=0
                tmp.append(val)
            mat.append(tmp)
            samples.append(k)
        
        
    elif chrom !="all":
        chrom_size=chrom_sizes[chrom]
        samples=list(files.keys())
        mat=Parallel(n_jobs=-1)(delayed(read_bw)(files[s],chrom, chrom_size, step) for s in samples)
        # for k, f in files.items():
        #     bw=pwg.open(f)
        #     val=np.array(bw.values(chrom,0,chrom_size))
        #     rem=chrom_size%step
        #     val=val[:val.shape[0]-rem]
        #     print(val.shape)
        #     val=np.nan_to_num(val)
        #     val=val.reshape([-1,step]).mean(axis=1)
        #     print(val.shape)
        #     mat.append(val/np.sum(val))
        #     samples.append(k)
    elif chrom=="all":
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
    else:
        raise Exception("please specify chrom or peakfile options.")
    
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
    
def call_superenhancer(bigwig: str, 
                       peakfile: str,
                       tss: str="",
                       stitch=25000,
                       tss_dist: int=5000):
    
    
    bw=pwg.open(bigwig)
    chrom_sizes=bw.chroms()
    peaks=read_peaks(peakfile)
    stitched=stitching(peaks, stitch)
    if tss != "":
        tss_pos=read_tss(tss,tss_dist)
        stitched=remove_close_to_tss(stitched, tss_pos)
    
    mat=[]
    pos=[]
    for chrom, se in stitched.items():
        for s, e in se:
            #print(chrom,s,e)
            val=bw.stats(chrom,s,e, type="sum", exact=True)[0]
            if val==None:
                val=0
            mat.append(val)
            pos.append(chrom+":"+str(s)+"-"+str(e))
            
    x, y, pos, sindex=find_extremes(mat, pos)
    b=-x[sindex]*np.amax(y) +y[sindex]
    fig, ax=plt.subplots()
    ax.scatter(x[:sindex], y[:sindex],s=5,rasterized=True)
    ax.scatter(x[sindex:], y[sindex:], color="r",s=5,rasterized=True)
    ax.plot(x,x*np.amax(y)+b, color="gray")
    
    for j in range(5):
        ax.text(x[-j-1],y[-j-1], pos[-j-1])
    ax.text(0,y[-1]*0.7, "SE No.: "+str( y[sindex:].shape[0])+"\nCutoff: "+str(np.round(y[sindex],3)))
    ax.set_xlabel("Enhancers")
    ax.set_ylabel("Signal sum")
    ax.set_ylim(0,max(y)*1.05)
    ax.set_xticks(np.arange(0,x.shape[0], 5000)/x.shape[0],labels=np.arange(0,x.shape[0], 5000))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    stitched_out=os.path.splitext(peakfile)[0]+"_SE.bed"
    with open(stitched_out, "w") as fout:
        rank=1
        for pos, sig in zip(reversed(pos[sindex:]), reversed(y[sindex:])):
            chrom, se=pos.split(":")
            s, e=se.split("-")
            fout.write("\t".join([chrom,str(s),str(e), str(sig),str(rank)+"\n"]))
            rank+=1
def plot_average():
    pass


if __name__=="__main__":
    #test="plot_bigwig"
    test="plot_bigwig_correlation"
    test="call_superenhancer"
    import glob
    if test=="plot_bigwig":
        fs= {"KMT2A":"/media/koh/grasnas/home/data/omniplot/HepG2_KMT2A-human_ENCFF406SHU.bw",
             "KMT2B":"/media/koh/grasnas/home/data/omniplot/HepG2_KMT2B-human_ENCFF709UTL.bw"}
        bed="/media/koh/grasnas/home/data/omniplot/tmp3.bed"
        gff="/media/koh/grasnas/home/data/omniplot/gencode.v40.annotation.gff3"
        
        plot_bigwig(fs,bed,gff, step=10)
        plt.show()
    elif test=="plot_bigwig_correlation":
        fs= {"KMT2A":"/media/koh/grasnas/home/data/omniplot/HepG2_KMT2A-human_ENCFF406SHU.bw",
             "KMT2B":"/media/koh/grasnas/home/data/omniplot/HepG2_KMT2B-human_ENCFF709UTL.bw",
             "HNF1A_rep1":"/media/koh/grasnas/home/data/omniplot/HepG2_HNF1A-human_ENCFF397BTX.bw",
             "HNF1A_rep2":"/media/koh/grasnas/home/data/omniplot/HepG2_HNF1A-human_ENCFF502ACF.bw",
             "HNF4A_rep1":"/media/koh/grasnas/home/data/omniplot/HepG2_HNF4A-human_ENCFF467SMI.bw",
             "HNF4A_rep2":"/media/koh/grasnas/home/data/omniplot/HepG2_HNF4A-human_ENCFF712VYA.bw",}
        
        plot_bigwig_correlation(fs,step=1000)
        plt.show()
    elif test=="call_superenhancer":
        f="/media/koh/grasnas/home/data/omniplot/HepG2_HNF1A-human_ENCFF397BTX.bw"
        peak="/media/koh/grasnas/home/data/omniplot/HNF1A_ENCFF696TGC_srt.bed"
        tss="/media/koh/grasnas/home/data/omniplot/gencode.v40.annotation_tss_srt.bed"
        call_superenhancer(bigwig=f, peakfile=peak,tss=tss)
        plt.show()
    #raise NotImplementedError("This function will plot ChIP-seq data.")