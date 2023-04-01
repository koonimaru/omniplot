from typing import List,Dict,Optional,Union,Any,Iterable
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os 
import scipy.stats
from scipy.spatial.distance import squareform, pdist
from scipy.spatial import distance
from joblib import Parallel, delayed
import itertools as it
from joblib.externals.loky import get_reusable_executor
from omniplot.chipseq_utils import (_stitching_for_pyrange, 
                                    gff_parser, 
                                    _read_peaks, 
                                    _read_bw, 
                                    _calc_pearson,
                                    _stitching,
                                    _read_tss,
                                    _remove_close_to_tss,
                                    _find_extremes,
                                    _readgff,
                                    _readgff2,
                                    _readgtf,
                                    _readgtf2,
                                    _read_and_reshape_bw,
                                    _read_bw_stats,
                                    _readgff_transcripts,
                                    _read_transcripts)
sns.set_theme(font="Arial", style={'grid.linestyle': "",'axes.facecolor': 'whitesmoke'})
import random
import string
from matplotlib.colors import LogNorm
from sklearn.decomposition import PCA, NMF, LatentDirichletAllocation
from scipy.stats import fisher_exact
from scipy.stats import zscore
from sklearn.cluster import KMeans
from omniplot.utils import _optimal_kmeans, _save
import time
import pyranges as pr
import warnings
import sys

try:
    import pyBigWig as pwg
except ImportError as e:
    print("If you want to use chipseq functions. You need to install pyBigWig.")
except Exception as err:
    print(f"Unexpected {err=}, {type(err)=}")
    raise

def plot_bigwig(files: dict, 
                bed: Union[str, list], 
                gff: str,
                step: int=100,
                stack_regions: str="horizontal", 
                highlight: Union[str, list]=[],
                highlightname="",
                n_jobs=-1,
                save: str="")->dict:
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
    for k, v in files.items():
        assert type(v)==str, "filenames must be str, but {} was give.".format(type(v))
        if not os.path.isfile(v)==True:
            raise Exception("{} does not exist.".format(v))
    if type(bed)==str:
        posrange=[]
        pos=[]
        with open(bed) as fin:
            
            for l in fin:
                l=l.split()
                chrom, s, e=l[0],l[1],l[2]
                posrange.append(pr.from_dict({"Chromosome": [chrom], "Start": [int(s.replace(",",""))], "End": [int(e.replace(",",""))]}))
                pos.append([chrom,int(s.replace(",","")),int(e.replace(",",""))])  
    else:
        pos=bed
        
    if type(highlight)==str:
        _highlight=[]
        with open(bed) as fin:
            for l in fin:
                l=l.split()
                chrom, s, e=l[0],l[1],l[2]
                _highlight.append([chrom,int(s.replace(",","")),int(e.replace(",",""))])
        highlight=_highlight
    
    time_start=time.time()
    if gff.endswith("gff") or gff.endswith("gff3"):
        gff_dict=_readgff2(gff, "gene", others=["gene_name", "Strand"])
    elif gff.endswith("gtf"):
        gff_dict=_readgtf2(gff, "transcript", others=["gene_name", "Strand"])
    else:
        raise Exception("The gff/gtf file name is required to have either gff, gff3, or gtf extension.")
    gff_ob=pr.from_dict(gff_dict)
    
    #
    # if gff.endswith("gff") or gff.endswith("gff3"):
    #     gff_ob=pr.read_gff3(gff)
    #     gff_ob=gff_ob[gff_ob.Feature=="gene"]
    #     gffformat="gff"
    # elif gff.endswith("gtf"):
    #     gff_ob=pr.read_gtf(gff)
    #     gff_ob=gff_ob[gff_ob.Feature=="transcript"]
    #     gffformat="gtf"
    print("Reading the gene annotation file took", time.time()-time_start)
    # geneset=[]
    # for chrom, start, end in pos:
    #     geneset.append(gff_ob.get_genes(chrom, start, end))
    time_start=time.time()
    geneset=[]
    _grs=Parallel(n_jobs=n_jobs)(delayed(gff_ob.overlap)(gr) for gr in posrange)
    # for gr in posrange:
    #     gr2=gff_ob.intersect(gr)
    for gr2 in _grs:
        #gr2=gff_ob.intersect(gr)
        if len(gr2)==0:
            geneset.append([])
        else:
            #print(gr2)
            # if gffformat=="gff":
            #     gr2=gr2[gr2.Feature=="gene"]
            # elif gffformat=="gtf":
            #     gr2=gr2[gr2.Feature=="transcript"]
                
            tmp=[]
            for gene_name, start, end, ori in zip(gr2.gene_name, gr2.Start, gr2.End, gr2.Strand):
                tmp.append([gene_name, start, end, ori])
            geneset.append(tmp)
    #print(geneset)
    #geneset=Parallel(n_jobs=-1)(delayed(gff_ob.get_genes)(chrom, start, end) for chrom, start, end in pos)
    print("Obtaining genes took", time.time()-time_start)
            
    time_start=time.time()
    mat={}
    for samplename, f in files.items():
        bw=pwg.open(f)
        #h, t=os.path.split(f)
        #samplename=t.replace(prefix, "").replace(suffix_set, "")
        mat[samplename]=[]
        
        for chrom, s, e in pos:
            val=bw.values(chrom, s, e)
            
            mat[samplename].append(val)
    
    
    
    print("Reading bigwig files took", time.time()-time_start)
    
    if stack_regions=="vertical":
        fig, axes=plt.subplots(nrows=len(pos)*2,ncols=len(files),figsize=[10,10],gridspec_kw={
                           'height_ratios': [2,1]*len(pos),"hspace":0},
                       constrained_layout = True)
        if len(axes.shape)==1:
            axes=axes.reshape([-1,1])
        sample_order=sorted(mat.keys())
        ticks_labels=[]
        for si, sample in enumerate(sample_order):
            
            vals=mat[sample]
            ymax=0
            for posi, (genes, val) in enumerate(zip(geneset, vals)):
                if len(val)==0:
                    continue
                ax=axes[posi*2,si]
                chrom, s, e=pos[posi]
                val=np.array(val)
                vallen=val.shape[0]
                remains=vallen%step
                _e=e-remains
                x=np.arange(s, _e,step)
                #print(s, e, _e, vallen, step)
                val=val[:vallen-remains].reshape(-1, step).mean(axis=1)
                _ymax=np.amax(val)
                if _ymax >  ymax:
                    ymax =_ymax
                if len(highlight)!=0:
                    hchrom, hs, he=highlight[posi]
                    ax.fill_between([hs, he], [_ymax, _ymax], alpha=0.5, color="r")
                ax.fill_between(x, val,edgecolor='face')
                
                
                if si==0:
                    ax.set_title(sample+" "+chrom+":"+str(s)+"-"+str(e))
                    ticks_labels.append([[np.amin(x),np.amax(x)],ax.get_xticks(), ax.get_xticklabels()])
                ax.set_xticks([],labels=[])
            occupied=[[] for i in range(8)]
        
        

        
        for posi in range(len(pos)):
            interval=pos[posi][2]-pos[posi][1]
            axes[posi*2+1,si].plot(ticks_labels[posi][0],[0,0],alpha=0)
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
                axes[posi*2+1,si].plot([gs, ge], [slot*1,slot*1], color="gray")
                if ori=="+":
                    axes[posi*2+1,si].plot([ge-interval/32, ge], [slot*1+1,slot*1], color="gray")
                elif ori=="-":
                    axes[posi*2+1,si].plot([gs, gs+interval/32], [slot*1,slot*1+1], color="gray")
                
                axes[posi*2+1,si].text((gs+ge)/2, slot*1, genename, ha="center")
            for oi, ol in enumerate(occupied):
                if len(ol)==0:
                    maxslot=oi
                    break
            axes[posi*2+1,si].set_ylim(-0.5,maxslot+0.5)
            #axes[posi*2+1,si].ticklabel_format(useOffset=False)
            axes[posi*2+1,si].set_yticks([],labels=[])
            axes[posi*2+1,si].set_xticks(ticks_labels[posi][1])
            axes[posi*2+1,si].set_ylabel("Genes")
        if save !="":
            if save.endswith(".pdf") or save.endswith(".png") or save.endswith(".svg"):
                h, t=os.path.splitext(save)
                plt.savefig(h+"_plot"+t)
            else:
                plt.savefig(save+"_plot.pdf")
    else:
        fig, axes=plt.subplots(nrows=len(files)+1,ncols=len(pos),gridspec_kw={
                           'height_ratios': [2]*len(files)+[1]})
        
        if len(axes.shape)==1:
            axes=axes.reshape([-1,1])
        sample_order=sorted(mat.keys())
        for si, sample in enumerate(sample_order):
            vals=mat[sample]
            ymax=0
            #print(ymax)
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
                
                #print(s, e, _e, vallen, step)
                val=val[:vallen-remains].reshape(-1, step).mean(axis=1)
                _ymax=np.amax(val)
                if _ymax >  ymax:
                    ymax =_ymax

                ax.fill_between(x, val,edgecolor='face')

                if posi==0:
                    ax.set_ylabel(sample)
    
                ax.set_xticks([],labels=[])
                if si==0:
                    ax.set_title(chrom+":"+str(s)+"-"+str(e))
                ax.grid(False)
                    
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
                #axes[si+1,posi].grid(False)
            for oi, ol in enumerate(occupied):
                if len(ol)==0:
                    maxslot=oi
                    break
            axes[si+1,posi].set_ylim(-0.5,maxslot+0.5)
            axes[si+1,posi].ticklabel_format(useOffset=False)
            axes[si+1,posi].set_yticks([],labels=[])
            
        
        plt.tight_layout(h_pad=-1,w_pad=-1)
        _save(save, "plot")
        
    return {"ax":axes,"values":mat,"genes":geneset,"positions":pos}


def plot_bigwig_correlation(files: dict, 
                            chrom: str="chr1",
                            step: int=1000,
                            palette: str="coolwarm",
                            figsize: list=[6,6],
                            show_val: bool=True,
                            clustermap_param: dict={},
                            peakfile: str="",
                            #show_val: bool=False,
                            show: bool=False,
                            n_jobs: int=-1, save: str="") -> Dict:
    
    """
    Calculate pearson correlations and draw a heatmap for bigwig files.  
    
    Parameters
    ----------
    files : dict
        A dictionary whose keys are sample names and values are bigwig file names 
    chrom : str, optional
        A chromsome to cal
    step: int, optional
        A bin size to reduce the bigwig signal resolution.
    palette: str, optional
    show_val: bool, optional
        Whether or not to show correlation values on the heatmap
    peakfile: str, optional
        A peak file/bed file containing genome regions of interest. If given this option, the correlation will be calculated using the genome regions listed in this file.
    show : bool, optional
        Whether or not to show the figure.
    n_jobs : int, optional
        The number of threads to use. It will use all available threads by default. 
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
    for k, v in files.items():
        assert type(v)==str, "filenames must be str, but {} was give.".format(type(v))
        if not os.path.isfile(v)==True:
            raise Exception("{} does not exist.".format(v))
    
    mat=[]
    samples=[]
    bw=pwg.open(list(files.values())[0])
    chrom_sizes=bw.chroms()
    
    
        
    
    if peakfile!="":
        # peaks=read_peaks(peakfile)
        # for k, f in files.items():
        #     bw=pwg.open(f)
        #     tmp=[]
        #     for chrom, se in peaks.items():
        #         for s, e in se:
        #             val=bw.stats(chrom,s,e, exact=True)[0]
        #             if val==None:
        #                 val=0
        #         tmp.append(val)
        #     mat.append(tmp)
        #     samples.append(k)
        peaks=_read_peaks(peakfile)
        for k, f in files.items():
            bw=pwg.open(f)
            
            tmp=Parallel(n_jobs=n_jobs)(delayed(_read_bw_stats)(chrom, s, e, f) for chrom, s, e in peaks)
            mat.append(tmp)
            samples.append(k)
        
    elif chrom !="all":
        chrom_size=chrom_sizes[chrom]
        
        samples=[]
        mat=[]
        #samples=list(files.keys())
        #mat=Parallel(n_jobs=n_jobs)(delayed(read_bw)(files[s],chrom, chrom_size, step) for s in samples)
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
    elif chrom=="all":
        chroms=chrom_sizes.keys()
        for k, f in files.items():
            
            #tmp=Parallel(n_jobs=n_jobs)(delayed(read_bw)(f, chrom, 0, chrom_sizes[chrom], f) for chrom, s, e in chroms)
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
    dmat=Parallel(n_jobs=-1)(delayed(_calc_pearson)(ind, mat) for ind in list(it.combinations(range(mat.shape[0]), 2)))
    dmat=np.array(dmat)
    get_reusable_executor().shutdown(wait=True)
    dmat=squareform(dmat)
    print(dmat)
    dmat+=np.identity(dmat.shape[0])
    g=sns.clustermap(data=dmat,xticklabels=samples,yticklabels=samples,
               method="ward", cmap=palette,
               col_cluster=True,
               row_cluster=True,
               figsize=figsize,
               rasterized=True,
               #cbar_kws={"label":"Pearson correlation"}, 
               annot=show_val,
               **clustermap_param)
    g.cax.set_ylabel("Pearson correlation", rotation=-90,va="bottom")
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)  # For y axis
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90) # For x axis
    
    
    _save(save, "correlation")
    if show:
        plt.show()
    return {"correlation": dmat,"samples":samples}
    
def call_superenhancer(bigwig: str, 
                       bed: str,
                       stitch=25000,
                       tss_dist: int=0,
                       plot_signals=False,
                       gff: str="",
                       closest_genes: bool=False,
                       nearest_k: int=1,
                       go_analysis: bool=False,
                       gonames: list=["GO_Biological_Process_2021","Reactome_2022","WikiPathways_2019_Human"],
                       sample: str="Sample",
                       n_jobs: int=12,
                       save: str="")->Dict:
    """
    Find super enhancers and plot an enhancer rank.  
    
    Parameters
    ----------
    bigwig : str
        A bigwig file.
    peakfile : str
        A peak file.
    tss_dist: int, optional
        A distance from transcriptional start sites. If this option is more than 0, you need provide gff option too.
    stitch: int, optional
        The maximum distance of enhancers to stitch together.
    plot_signals : bool, optional
        Drawing super enhancer regions +- 10kb
    gff: str, optional (required when tss_dist is provided)
        A gff or gtf file that contains gene annotations. You need to chose a correct genome version of a gff/gtf file 
    
    closest_genes: bool=False,
        If this is True, closest genes of SE will be shown, instead of genomic regions. You can specify the number 
        of closest genes by "nearest_k" option.
    nearest_k: int, optional
        The number of closest genes to be included. Default: 1.
        
    go_analysis: bool, optional
        GSEA analysis based on the rank of SEs.
        
    gonames: list=["GO_Biological_Process_2021","Reactome_2022","WikiPathways_2019_Human"]
        Gene sets for GSEA analysis.
    
    sample: str,optional
        The sample name to appear in the signal plot
    
    Returns
    -------
        dict {"ax": ax, "positions":pos, "signals":y,"nearest_genes":_nearestgenes}
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
    
    bw=pwg.open(bigwig)
    chrom_sizes=bw.chroms()
    peaks=_read_peaks(bed)
    #stitched=stitching(peaks, stitch)
    
    
    if tss_dist != 0:
        start_time=time.time()
        stitched=pr.from_dict(_stitching_for_pyrange(peaks, stitch))
        
        if gff.endswith("gff") or gff.endswith("gff3"):
            gff_dict=_readgff(gff, "transcript",tss_dist)
        elif gff.endswith("gtf"):
            gff_dict=_readgtf(gff, "transcript",tss_dist)
        else:
            raise Exception("The gff/gtf file name is required to have either gff, gff3, or gtf extension.")
        gffr=pr.from_dict(gff_dict)
        
        _stitched=stitched.subtract(gffr, nb_cpu=n_jobs)
        stitched={}
        for chrom, s, e in zip(_stitched.Chromosome, _stitched.Start, _stitched.End):
            if not chrom in stitched:
                stitched[chrom]=[]
            stitched[chrom].append([s,e])
        print("removing enhancers around TSS took ", time.time()-start_time)
        # tss_pos=read_tss(tss,tss_dist)
        # stitched=remove_close_to_tss(stitched, tss_pos)
    else:
        stitched=_stitching(peaks, stitch)
        
        
    """Obtaining closest genes"""
    _nearestgenes=[]
    if closest_genes==True:
        start_time=time.time()
        if gff.endswith("gff") or gff.endswith("gff3"):
            gff_dict=_readgff2(gff, "gene", others=["gene_name"])
        elif gff.endswith("gtf"):
            gff_dict=_readgtf2(gff, "transcript", others=["gene_name"])
        else:
            raise Exception("The gff/gtf file name is required to have either gff, gff3, or gtf extension.")
        gffr=pr.from_dict(gff_dict)
        # grlist=[]
        # for chrom_se in pos:
        #     chrom, se=chrom_se.split(":")
        #     s, e=se.split("-")
        #     grlist.append(pr.from_dict({"Chromosome":[chrom],"Start":[int(s)],"End":[int(e)]}))
        # if nearest_k==1:
        #     _nearestgenes=Parallel(n_jobs=-1)(delayed(gr.nearest)(gffr) for gr in grlist)
        # else:
        #     _nearestgenes=Parallel(n_jobs=-1)(delayed(gr.k_nearest)(gffr, k=nearest_k) for gr in grlist)
        grlist={"Chromosome":[],"Start":[],"End":[]}
        stiched_dit={}
        for chrom, se in stitched.items():
            for s, e in se:
                stiched_dit[chrom+":"+str(s)+"-"+str(e)]=[]
                #grlist.append(pr.from_dict({"Chromosome":[chrom],"Start":[int(s)],"End":[int(e)]}))
                grlist["Chromosome"].append(chrom)
                grlist["Start"].append(int(s))
                grlist["End"].append(int(e))
                
        gr=pr.from_dict(grlist)
        _nearestgenes_gr=gr.k_nearest(gffr, k=nearest_k, nb_cpu=n_jobs)
        # _nearestgenes=[]
        # tmp=[]
        for i, (chrom, s, e, gene_name) in enumerate(zip(_nearestgenes_gr.Chromosome, _nearestgenes_gr.Start,_nearestgenes_gr.End,_nearestgenes_gr.gene_name)):
            stiched_dit[chrom+":"+str(s)+"-"+str(e)].append(gene_name)
        #
        #     tmp.append(gene_name)
        #     if i%nearest_k==0 and i>0:
        #         _nearestgenes.append(tmp)
        #         tmp=[]
        # _nearestgenes.append(tmp)
        print("Finding the nearest genes took ", time.time()-start_time)
        
    start_time=time.time()
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
            
    x, y, pos, sindex, srt=_find_extremes(mat, pos)
    b=-x[sindex]*np.amax(y) +y[sindex]
    print("Finding super enhancer took ", time.time()-start_time, "sec")
    
    # print(len(_nearestgenes), _nearestgenes[:10])
    # print(srt.shape, srt[:10])
    _nearestgenes=[stiched_dit[chromse] for chromse in pos]
    
        
    fig, ax=plt.subplots(figsize=[6,4])
    ax.scatter(x[:sindex], y[:sindex],s=5,rasterized=True)
    ax.scatter(x[sindex:], y[sindex:], color="r",s=5,rasterized=True)
    ax.plot(x,x*np.amax(y)+b, color="gray")
    
    if closest_genes==True:
        for j in range(5):
            ax.text(x[-j-1],y[-j-1], ",".join(_nearestgenes[-j-1]))
    else:
        for j in range(5):
            ax.text(x[-j-1],y[-j-1], pos[-j-1])
    ax.text(0,y[-1]*0.7, "SE No.: "+str( y[sindex:].shape[0])+"\nCutoff: "+str(np.round(y[sindex],3)))
    ax.set_xlabel("Enhancers")
    ax.set_ylabel("Signal sum")
    ax.set_ylim(0,max(y)*1.05)
    
    tick_=x.shape[0]//3
    if 100<tick_<=1000:
        tick_=500
    elif 1000<tick_<=5000:
        tick_=1000
    elif 5000<tick_<=10000:
        tick_=5000
    else:
        tick_=10000
    ax.set_xticks(np.arange(0,x.shape[0], tick_)/x.shape[0],labels=np.arange(0,x.shape[0], tick_))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    
    _save(save, "rank")
    
    plt.subplots_adjust(right=0.620, bottom=0.130)
    
    
    stitched_out=os.path.splitext(bed)[0]+"_SE.bed"
    with open(stitched_out, "w") as fout:
        
        if closest_genes==True:
            rank=1
            for _pos, _sig, _gr in zip(reversed(pos[sindex:]), reversed(y[sindex:]),reversed(_nearestgenes[sindex:])):
                chrom, se=_pos.split(":")
                s, e=se.split("-")
                fout.write("\t".join([chrom,str(s),str(e), ",".join(_gr), str(_sig),str(rank)+"\n"]))
                rank+=1
        
        else:
            rank=1
            for _pos, _sig in zip(reversed(pos[sindex:]), reversed(y[sindex:])):
                chrom, se=_pos.split(":")
                s, e=se.split("-")
                fout.write("\t".join([chrom,str(s),str(e), str(_sig),str(rank)+"\n"]))
                rank+=1
    print("The SE file was written in "+stitched_out)
    if go_analysis==True:
        print("Now, doing GO anlyses..")
        start_time=time.time()
        try:
            import gseapy
        except ImportError:
            raise ImportError('If you want GO analysis, you need to install gseapy. Try "pip install gseapy".')
        godf={0:[],1:[]}
        y=(y-np.mean(y))/np.std(y)
        for _sig, genes in zip(reversed(y), reversed(_nearestgenes)):
            for gene in list(genes):
                godf[0].append(gene)
                godf[1].append(_sig)
        #print(godf)
        godf=pd.DataFrame(godf)
        godf=godf.set_index(0)
        for name in gonames:
            gs_res = gseapy.prerank(rnk=godf, # or data='./P53_resampling_data.txt'
                         gene_sets=name,
                             threads=4,
                             min_size=5,
                             max_size=1000,
                             permutation_num=100, # reduce number to speed up testing
                             outdir=None, # don't write to disk
                             seed=6,
                             verbose=True,) # see what's going on behind the scenes)
            #print(gs_res.res2d)
            for index, row in gs_res.res2d.iterrows():
                #print(row)
                if row["FDR q-val"] <0.05:
                    t=row.Term
                    gseapy.gseaplot(rank_metric=gs_res.ranking,
                             term=t,
                             cmap=plt.cm.seismic,
                             **gs_res.results[t],
                             ofname=os.path.splitext(bed)[0]+"_"+name+"_gsea.pdf"
                             )
        print("GO analysis took ", time.time()-start_time)
    if plot_signals:
        
        if gff=="":
            raise Exception("Please provide a gff file to draw signals.")
        letters = string.ascii_lowercase
        tmpfile=''.join(random.choice(letters) for i in range(10))+".bed"
        highlights=[]
        with open(tmpfile, "w") as fout:
            rank=1
            for pos, sig in zip(reversed(pos[sindex:]), reversed(y[sindex:])):
                chrom, se=pos.split(":")
                s, e=se.split("-")
                fout.write("\t".join([chrom,str(int(s)-10000),str(int(e)+10000), str(sig),str(rank)+"\n"]))
                highlights.append([chrom,int(s),int(e)])
                rank+=1
                if rank==6:
                    break
                    
        plot_bigwig(files={sample: bigwig}, 
                    bed=tmpfile, gff=gff, 
                    step=100,
                    stack_regions="vertical",
                    highlight=highlights,n_jobs=n_jobs, save=save)
        os.remove(tmpfile)
    return {"ax": ax, "positions":pos, "signals":y,"nearest_genes":_nearestgenes}


def plot_average(files: dict, 
                 bed: Union[str,list], 
                 order: list=[], extend: int=500, 
                 palette: str="coolwarm",
                 binsize: int=10,
                 clustering: str="",
                 n_clusters: int=5,
                 minval: Optional[float]=None,
                 orientaion=False,
                 save="",
                 n_jobs: int=-1):
    """
    Plotting bigwig files centered at peak regions.  
    
    Parameters
    ----------
    files : dict
        A dictionary whose keys are sample names and values are file names 
    bed : str
        A bed file name containing the list of genomic regions
    order : list
        A sample order to plot. if a clustering option (except kmeans_all) is chosen, the first sample is the target of clustering analysis. 
    binsize: int
        A bin size to reduce the bigwig signal resolution.
    clustering : str
        Availables: ["kmeans", "kmeans_all", "kmeans_auto"]
    n_clusters : int
        the number of clusters for clustering analysis. if you chose kmeans_auto, this option will be ignored.
    Returns
    -------
        dict {"values":data,"potisons":pos,"labels":labels}
        values: dictionary containing sample names as the keys and signals as the values
        positions: genomic positions sorted by signals displayed in the plot
        labels: list of clustering labels if kmean clustering was performed
        
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
    if len(order)==0:
        order=list(files.keys())
    for k, v in files.items():
        assert type(v)==str, "filenames must be str, but {} was give.".format(type(v))
        if not os.path.isfile(v)==True:
            raise Exception("{} does not exist.".format(v))
            
    
    
    bw=pwg.open(files[order[0]])
    chrom_sizes=bw.chroms()
    
    if type(bed)==str:
        
        pos=[]
        bedchroms=set()
        with open(bed) as fin:
            for l in fin:
                if l.startswith("#"):
                    continue
                l=l.split()
                chrom, s, e=l[0],l[1],l[2]
                
                if not chrom in chrom_sizes:
                    raise Exception("There may be mismatches in chromosome names between the bed file and bigwig files.\n\
                the chromosome names in the bed file are: {}\n\
                those in the bigwig files are: {}".format(chrom, chrom_sizes.keys())) 
                
                s, e=int(s.replace(",","")),int(e.replace(",",""))
                center=(s+e)//2
                cs=center-extend
                ce=center+extend
                if cs <0 or chrom_sizes[chrom] < ce:
                    continue
                pos.append([chrom,cs,ce])
                bedchroms.add(chrom)
    else:
        pos=bed
    
    
    data={}
    data_mean=[]
    
    for i, sample in enumerate(order):
        bigwig=files[sample]
        data[sample]=[]
        vals, mean=zip(*Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_read_and_reshape_bw)(chrom, s, e, bigwig, binsize) for chrom, s, e in pos))
        
        
        data[sample]=np.nan_to_num(vals)
        if i==0:
            data_mean=mean
        # for chrom, s, e in pos:
        #
        #     val=bw.values(chrom, s, e)
        #     #print(val)
        #     val=np.array(val)
        #     #print(val.shape)
        #     val=val.reshape([-1,binsize]).mean(axis=1)
        #     #print(val.shape)
        #     data[sample].append(val)
        #     if i==0:
        #         data_mean.append(bw.stats(chrom, s, e, exact=True)[0])
    labels=[]
    if clustering=="kmeans_all":
        mat=[]
        for sample in order:
            #print(np.array(data[sample]).shape)
            mat.append(data[sample])
        mat=np.concatenate(mat, axis=1)
        #print(mat.shape)
        mat=zscore(mat, axis=0)
        pca=PCA(n_components=5, random_state=1)
        xpca=pca.fit_transform(mat)
        kmean = KMeans(n_clusters=n_clusters, random_state=0,n_init=10)
        kmX=kmean.fit(xpca)
        labels=kmX.labels_
    elif clustering=="kmeans":
        mat=np.array(data[order[0]])
        kmean = KMeans(n_clusters=n_clusters, random_state=0,n_init=10)
        kmX=kmean.fit(mat)
        labels=kmX.labels_
    elif clustering=="kmeans_auto":
        mat=np.array(data[order[0]])
        optimalclusternum=_optimal_kmeans(mat, [2, 10])
        n_clusters=np.amax(optimalclusternum)
        kmean = KMeans(n_clusters=n_clusters, random_state=0,n_init=10)
        kmX=kmean.fit(mat)
        labels=kmX.labels_
        
    sortindex=np.argsort(data_mean)[::-1]
    pos=np.array(pos)[sortindex]

    plotindex=np.arange(0, len(pos), len(pos)//1000)
    fig, axes=plt.subplots(ncols=len(order), figsize=[2*len(order), 6])
    if len(order)==1:
        axes=[axes]
    for i, (sample, ax) in enumerate(zip(order, axes)):
        vals=data[sample]

        vals=np.array(vals)

        vals=vals[sortindex]
        if clustering!="":
            
            _labels=np.array(labels)[sortindex]
            sortindex2=np.argsort(_labels)
            vals=vals[sortindex2]
            _pos=pos[sortindex2]
            _labels=_labels[sortindex2]
            _labels=_labels[plotindex]
            ulabel, clabel=np.unique(_labels, return_counts=True)

        vals=vals[plotindex]
        if np.sum(vals)<=10**-6:
            raise Exception("Signals may be empty, or you may be using a wrong bed file.")
        if np.amax(vals) <= 1  and np.amin(vals)>=0:
            im=ax.imshow(vals,aspect="auto", cmap=palette, interpolation="none",norm=LogNorm(vmin=0.00001))
        elif np.amin(vals) < 0:
            vals=np.where(vals < 0, 0, vals)
            warnings.warn("The bigwig contains negative values. Is this coverage file?")
            im=ax.imshow(vals,aspect="auto", cmap=palette, interpolation="none",norm=LogNorm(vmin=1))
        elif minval!=None:
            im=ax.imshow(vals,aspect="auto", cmap=palette, interpolation="none",norm=LogNorm(vmin=minval))
        else:
            im=ax.imshow(vals,aspect="auto", cmap=palette, interpolation="none",norm=LogNorm(vmin=1))
        
        
        
        cbaxes = fig.add_axes([0.8*(i+1)/len(order), 0.93, 0.03, 0.06]) 
        cb = plt.colorbar(im, cax = cbaxes)
        
            
        # print(np.arange(0, vals.shape[1]+vals.shape[1]//4, vals.shape[1]//4))
        # print(np.arange(-extend, extend+extend//2, extend//2))
        ax.set_xticks(np.arange(0, vals.shape[1]+vals.shape[1]//4, vals.shape[1]//4),labels=np.arange(-extend, extend+extend//2, extend//2))
        ax.set_xlabel("Peak range [bp]")
        ax.set_title(sample)
        
        if clustering!="":
            total=0
            isep=0
            for label, count in zip(ulabel, clabel):
                
                total+=count
                if isep==ulabel.shape[0]-1:
                    break
                ax.plot([0, vals.shape[1]-1], [total,total], color="black")
                isep+=1
        ax.set_yticks([])
        ax.grid(False)
    _save(save, "heatmap")
    fig, axes2=plt.subplots(ncols=len(order), figsize=[4*len(order), 3])
    if len(order)==1:
        axes2=[axes2]
    maxval=0
    for i, (sample, ax) in enumerate(zip(order, axes2)):
        vals=data[sample]
        vals=np.array(vals)
        
        labels=np.array(labels)
        if clustering !="":
            for label, count in zip(ulabel, clabel):
                
                _vals=np.mean(vals[labels==label],axis=0)
                #print(_vals.shape)
                #print(vals.shape)
                ax.plot(np.arange(-extend, extend, 2*extend//(_vals.shape[0])),_vals, label=label)
                _maxval=np.max(_vals)
                if maxval < _maxval:
                    maxval=_maxval
            plt.legend()
        else:
            vals=vals.mean(axis=0)
            ax.plot(np.arange(-extend, extend, 2*extend//(vals.shape[0])),vals)
            _maxval=np.max(vals)
            if maxval < _maxval:
                maxval=_maxval 
        if i==0:
            ax.set_ylabel("Mean signal value")
        # print(np.arange(0, vals.shape[1]+vals.shape[1]//4, vals.shape[1]//4))
        # print(np.arange(-extend, extend+extend//2, extend//2))
        #ax.set_xticks(np.arange(0, vals.shape[1]+vals.shape[1]//4, vals.shape[1]//4),labels=np.arange(-extend, extend+extend//2, extend//2))
        ax.set_xlabel("Peak range [bp]")
        ax.set_title(sample)
    
    for ax in axes2:
        ax.set_ylim(0, maxval*1.05)
    
    _save(save, "average")
    
    return {"values":data,"potisons":_pos,"labels":labels, "axes":axes,"axes2":axes2}



def _plot_genebody(files: dict,
                files_minus: Optional[dict]=None,
                 bed: Union[list, str]="",
                 gff: str="",
                 centered_at: str="TSS", 
                 order: list=[], 
                 extend: int=3000, 
                 palette: str="coolwarm",
                 binsize: int=10,
                 clustering: str="",
                 n_clusters: int=5,
                 save: str="",lognorm=True,n_jobs=-1)-> Dict:
    
    """
    Plotting bigwig files around gene bodies.  
    
    Parameters
    ----------
    files : dict
        A dictionary whose keys are sample names and values are file names 
    bed : str
        A bed file name containing the list of genomic regions
    order : list
        A sample order to plot. if a clustering option (except kmeans_all) is chosen, the first sample is the target of clustering analysis. 
    binsize: int
        A bin size to reduce the bigwig signal resolution.
    clustering : str
        Availables: ["kmeans", "kmeans_all", "kmeans_auto"]
    n_clusters : int
        the number of clusters for clustering analysis. if you chose kmeans_auto, this option will be ignored.
    Returns
    -------
        dict {"values":data,"potisons":pos,"labels":labels, "axes":axes, "axes2":axes2}
        values: dictionary containing sample names as the keys and signals as the values
        positions: genomic positions sorted by signals displayed in the plot
        labels: list of clustering labels if kmean clustering was performed
        
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
    if len(order)==0:
        order=list(files.keys())
    for k, v in files.items():
        assert type(v)==str, "filenames must be str, but {} was give.".format(type(v))
        if not os.path.isfile(v)==True:
            raise Exception("{} does not exist.".format(v))
    bw=pwg.open(files[order[0]])
    chrom_sizes=bw.chroms()
    
    if len(bed)>0 and len(gff)>0:
        warnings.warn("Ignoring the gff file, since the bed file is provided.")
    
    if len(bed)>0:
        oriset=set(["+", "-"])
        bedchroms=set()
        if type(bed)==str and len(bed)>0:
            with open(bed) as fin:
                for l in fin:
                    l=l.split()
                    if not l[-1] in oriset:
                        raise Exception("The bed file must contain orientation information at the last column. e.g., chr1\t10000\t20000\t.\t.\t+")
                    
                    if not l[0] in chrom_sizes:
                        raise Exception("There may be mismatches in chromosome names between the bed file and bigwig files.\n\
                    the chromosome names in the bed file are: {}\n\
                    those in the bigwig files are: {}".format(l[0], chrom_sizes.keys())) 
            posplus=[]
            posminus=[]
            seen=set()
            with open(bed) as fin:
                for l in fin:
                    l=l.split()
                    chrom, s, e, ori=l[0],l[1],l[2],l[-1]
                    tmp="_".join([chrom, s, e, ori])
                    if not tmp in seen:
                        seen.add(tmp)
                    else:
                        continue
                    
                    bedchroms.add(chrom)
                    if ori=="+":
                        cs=int(s)-extend
                        ce=int(s)+extend
                        if cs <0 or chrom_sizes[chrom] < ce:
                            continue
                        posplus.append([chrom,cs,ce])
                    else:
                        cs=int(e)-extend
                        ce=int(e)+extend
                        if cs <0 or chrom_sizes[chrom] < ce:
                            continue
    
                        posminus.append([chrom,cs,ce])
        else:
            posplus=bed[0]
            posminus=bed[1]
    
    elif len(gff)>0:
        transcripts=_readgff_transcripts(gff, extend)
    
    
    if len(order)==0:
        order=list(files.keys())
    data={}
    data_mean=[]
    start_time=time.time()
    
    for i, sample in enumerate(order):
        bigwig=files[sample]
        
        if len(gff)>0:
            tid_list=[]
            pvals=[]
            tids=transcripts["+"].keys()
            tid_list.extend(tids)
            pvals=Parallel(n_jobs=n_jobs)(delayed(_read_transcripts)(bigwig, transcripts["+"][tid], binsize, extend)  for tid in tids)
            print(pvals[:10])
            if files_minus!=None:
                bigwig=files_minus[sample]
            mvals=[]
            tids=transcripts["-"].keys()
            mvals=Parallel(n_jobs=n_jobs)(delayed(_read_transcripts)(bigwig, transcripts["-"][tid], binsize, extend)  for tid in tids)
            tid_list.extend(tids)
            
        #bw=pwg.open(bigwig)
        else:
            pvals=[]
            pmean=[]
            data[sample]=[]
            pvals, pmean=zip(*Parallel(n_jobs=n_jobs)(delayed(_read_and_reshape_bw)(chrom, s, e, bigwig, binsize)  for chrom, s, e in posplus))
            
            if files_minus!=None:
                bigwig=files_minus[sample]
                #bw=pwg.open(bigwig)
                mvals, mmean=zip(*Parallel(n_jobs=n_jobs)(delayed(_read_and_reshape_bw)(chrom, s, e, bigwig, binsize)  for chrom, s, e in posminus))
            else:
                mvals, mmean=zip(*Parallel(n_jobs=n_jobs)(delayed(_read_and_reshape_bw)(chrom, s, e, bigwig, binsize)  for chrom, s, e in posminus))
        
        mvals=np.flip(mvals, axis=1)
        vals=np.nan_to_num(np.concatenate([pvals,mvals]))
        print(vals.shape)
        if i ==0:
            sums=np.sum(vals, axis=1)
            remove=sums >np.quantile(sums, 0.1)
            print(np.sum(remove))
        vals=vals[remove]
        #vals=vals.reshape([-1,vals.shape[1]//binsize,binsize]).mean(axis=2)
        print(np.amax(sums),np.amin(sums),vals.shape)
        data[sample]=vals
        
        if i==0:
            data_mean=np.amax(vals,axis=1)
        # for chrom, s, e in posplus:
        #     val=bw.values(chrom, s, e)
        #     #print(val)
        #     val=np.array(val)
        #     #print(val.shape)
        #     val=val.reshape([-1,binsize]).mean(axis=1)
        #     #print(val.shape)
        #     data[sample].append(val)
        #     if i==0:
        #         mval=bw.stats(chrom, s, e, exact=True)[0]
        #         if mval==None:
        #             data_mean.append(0)
        #         else:
        #             data_mean.append(mval)
        #
        # for chrom, s, e in posminus:
        #     chromsize=chrom_sizes[chrom]
        #     val=bw.values(chrom, s, e)
        #     #print(val)
        #     val=np.array(val)[::-1]
        #     #print(val.shape)
        #     val=val.reshape([-1,binsize]).mean(axis=1)
        #     #print(val.shape)
        #     data[sample].append(val)
        #     if i==0:
        #         mval=bw.stats(chrom, s, e, exact=True)[0]
        #         if mval==None:
        #             data_mean.append(0)
        #         else:
        #             data_mean.append(mval)
    print("Reading bigwig file took ", time.time()-start_time)
    labels=[]
    if clustering=="kmeans_all":
        mat=[]
        for sample in order:
            print(np.array(data[sample]).shape)
            mat.append(data[sample])
        mat=np.concatenate(mat, axis=1)
        mat=np.nan_to_num(mat)
        print(mat.shape)
        mat=zscore(mat, axis=0)
        pca=PCA(n_components=5, random_state=1)
        xpca=pca.fit_transform(mat)
        kmean = KMeans(n_clusters=n_clusters, random_state=0,n_init=10)
        kmX=kmean.fit(xpca)
        labels=kmX.labels_
    elif clustering=="kmeans":
        mat=np.nan_to_num(np.array(data[order[0]]))
        kmean = KMeans(n_clusters=n_clusters, random_state=0,n_init=10)
        kmX=kmean.fit(mat)
        labels=kmX.labels_
    elif clustering=="kmeans_auto":
        mat=np.nan_to_num(np.array(data[order[0]]))
        optimalclusternum=_optimal_kmeans(mat, [2, 10])
        n_clusters=np.amax(optimalclusternum)
        kmean = KMeans(n_clusters=n_clusters, random_state=0,n_init=10)
        kmX=kmean.fit(mat)
        labels=kmX.labels_
    if len(gff) >0:
        pos=np.array(tid_list)
    else:
        pos=np.array(posplus+posminus)[remove]
    #data_mean=np.nan_to_num(data_mean)
    sortindex=np.argsort(data_mean)[::-1]
    print(sortindex)
    pos=pos[sortindex]
    print(pos[:20])
    plotindex=np.arange(0, len(pos), len(pos)//1000)
    fig, axes=plt.subplots(ncols=len(order), figsize=[2*len(order), 6])
    if len(order)==1:
        axes=[axes]
    for i, (sample, ax) in enumerate(zip(order, axes)):
        vals=data[sample]
        
        vals=np.array(vals)
        
        vals=vals[sortindex]
        vals=vals/np.amax(vals, axis=1)[:,None]
        ax.plot(vals[0])
        ax.plot(vals[1])
        ax.plot(vals[2])
        plt.show()
        sys.exit()
        

        if clustering!="":
            
            _labels=np.array(labels)[sortindex]
            sortindex2=np.argsort(_labels)
            vals=vals[sortindex2]
            _pos=pos[sortindex2]
            _labels=_labels[sortindex2]
            _labels=_labels[plotindex]
            ulabel, clabel=np.unique(_labels, return_counts=True)

        vals=vals[plotindex]
        print(vals.shape)
        if lognorm==True:
            
            im=ax.imshow(vals,aspect="auto", cmap=palette, interpolation="none",norm=LogNorm(vmin=np.amax([1, np.quantile(sums,0.5)])))
        else:
            im=ax.imshow(vals,aspect="auto", cmap=palette, interpolation="none")
        cbaxes = fig.add_axes([0.8*(i+1)/len(order), 0.93, 0.03, 0.06]) 
        cb = plt.colorbar(im, cax = cbaxes)
        
            
        # print(np.arange(0, vals.shape[1]+vals.shape[1]//4, vals.shape[1]//4))
        # print(np.arange(-extend, extend+extend//2, extend//2))
        ax.set_xticks(np.arange(0, vals.shape[1]+vals.shape[1]//4, vals.shape[1]//4),labels=np.arange(-extend, extend+extend//2, extend//2))
        ax.set_xlabel("Peak range [bp]")
        ax.set_title(sample)
        
        if clustering!="":
            total=0
            isep=0
            for label, count in zip(ulabel, clabel):
                
                total+=count
                if isep==ulabel.shape[0]-1:
                    break
                ax.plot([0, vals.shape[1]-1], [total,total], color="black")
                isep+=1
        ax.set_yticks([])
        ax.grid(False)
        
    _save(save, "heatmap")

    fig, axes2=plt.subplots(ncols=len(order), figsize=[4*len(order), 3])
    if len(order)==1:
        axes2=[axes2]
    maxval=0
    for i, (sample, ax) in enumerate(zip(order, axes2)):
        vals=data[sample]
        vals=np.array(vals)
        labels=np.array(labels)
        if clustering !="":
            for label, count in zip(ulabel, clabel):
                
                _vals=np.mean(vals[labels==label],axis=0)
                print(_vals.shape)
                print(vals.shape)
                ax.plot(np.arange(-extend, extend, 2*extend//(_vals.shape[0])),_vals, label=label)
                _maxval=np.max(_vals)
                if maxval < _maxval:
                    maxval=_maxval
            plt.legend()
        else:
            vals=vals.mean(axis=0)
            ax.plot(np.arange(-extend, extend, 2*extend//(vals.shape[0])),vals)
            _maxval=np.max(vals)
            if maxval < _maxval:
                maxval=_maxval 
        if i==0:
            ax.set_ylabel("Mean signal value")
        # print(np.arange(0, vals.shape[1]+vals.shape[1]//4, vals.shape[1]//4))
        # print(np.arange(-extend, extend+extend//2, extend//2))
        #ax.set_xticks(np.arange(0, vals.shape[1]+vals.shape[1]//4, vals.shape[1]//4),labels=np.arange(-extend, extend+extend//2, extend//2))
        ax.set_xlabel("Peak range [bp]")
        ax.set_title(sample)
    
    for ax in axes:
        ax.set_ylim(0, maxval*1.05)
    _save(save, "average")
    return {"values":data,"potisons":pos,"labels":labels, "axes":axes, "axes2":axes2}

def plot_bed_correlation(files:dict,
                         method: str="pearson",
                         palette: str="coolwarm",
                         figsize: list=[6,6], 
                         show_val: bool=True, 
                         clustermap_param: dict={},
                         step: int=100,
                         show: bool=False):
    for k, v in files.items():
        assert type(v)==str, "filenames must be str, but {} was give.".format(type(v))
        if not os.path.isfile(v)==True:
            raise Exception("{} does not exist.".format(v))
    bedintervals={}
    minmax={}
    for sample, bedfile in files.items():
        bedintervals[sample]={}
        with open(bedfile) as fin:
            for l in fin:
                if l.startswith("#"):
                    continue
                l=l.split()
                chrom=l[0]
                s, e=int(l[1]), int(l[2])
                if not chrom in bedintervals[sample]:
                    bedintervals[sample][chrom]=[]
                if not chrom in minmax:
                    minmax[chrom]=[]
                bedintervals[sample][chrom].append([s,e])
                minmax[chrom].append(s)
                minmax[chrom].append(e)
    minmaxval={}
    for chrom, se in minmax.items():
        minmaxval[chrom]=[np.min(se), np.max(se)]
    
    mat=[]
    samples=[]
    for sample, chrom_intervals in bedintervals.items():
        print(sample)
        tmp=[]
        for chrom, intervals in chrom_intervals.items():
            tmp2=[0 for _ in range(minmaxval[chrom][0], minmaxval[chrom][1],step)]
            for s, e in intervals:
                s=s-minmaxval[chrom][0]
                e=e-minmaxval[chrom][0]
                for i in range(s, e,step):
                    tmp2[i//step]=1
            tmp.extend(tmp2)
        mat.append(tmp)
        samples.append(sample)
    mat=np.array(mat)
    mat = mat[:, np.any(mat, axis=0)]
    print(mat[:100,:100])
    if method=="pearson":
        #dmat=Parallel(n_jobs=-1)(delayed(_calc_pearson)(ind, mat) for ind in list(it.combinations(range(mat.shape[0]), 2)))
        dmat=np.corrcoef(mat)
        dmat=np.array(dmat)
        #dmat=squareform(dmat)
        print(dmat)
        #dmat+=np.identity(dmat.shape[0])
    else:
        dmat=squareform(pdist(mat, method))
    
            
    
    g=sns.clustermap(data=dmat,xticklabels=samples,yticklabels=samples,
               method="ward", cmap=palette,
               col_cluster=True,
               row_cluster=True,
               figsize=figsize,
               rasterized=True,
               #cbar_kws={"label":"Pearson correlation"}, 
               annot=show_val,
               **clustermap_param)
    if method=="pearson":
        title="Pearson correlation"
    else:
        title=method+" distance"
    g.cax.set_ylabel(title, rotation=-90,va="bottom")
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)  # For y axis
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90) # For x axis
    if show:
        plt.show()
        
    return {"correlation": dmat,"samples":samples}
    
    
    
    
def frip_calc(files:dict,):
    
    pass
    
    
    
    

if __name__=="__main__":
    #test="plot_bigwig"
    test="plot_bigwig_correlation"
    
    #test="plot_bigwig"
    test="plot_bed_correlation"
    
    test="plot_average"
    #test="plot_genebody"
    #test="call_superenhancer"
    #test="plot_genebody"
    import glob
    
    if test=="plot_bed_correlation":
        fs= {"KMT2A":"/media/koh/grasnas/home/data/omniplot/KMT2A_ENCFF644EPI_srt.bed",
             "KMT2B":"/media/koh/grasnas/home/data/omniplot/HepG2_KMT2B_ENCFF036WCY_srt.bed",
             "HNF4G_rep1":"/media/koh/grasnas/home/data/omniplot/HNF4G_ENCFF413UQM_srt.bed",
             "HNF4G_rep2":"/media/koh/grasnas/home/data/omniplot/HNF4G_ENCFF123FQW_srt.bed",
             "HNF1A_rep1":"/media/koh/grasnas/home/data/omniplot/HNF1A_ENCFF696TGC_srt.bed",
             "HNF1A_rep2":"/media/koh/grasnas/home/data/omniplot/HNF1A_ENCFF162SGM_srt.bed",}
        
        plot_bed_correlation(fs,step=5000, method="correlation")
        plt.show()
    elif test=="plot_bigwig":
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
        gff="/media/koh/grasnas/home/data/omniplot/gencode.v40.annotation.gff3"
        
        gff="/media/koh/grasnas/home/data/omniplot/hg38.refGene.gtf"
        gff="/media/koh/grasnas/home/data/omniplot/gencode.vM31.annotation.gff3"
        f="/media/koh/grasnas/home/data/omniplot/HepG2_KMT2B-human_ENCFF709UTL.bw"
        f="/media/koh/grasnas/home/data/omniplot/SRR620141_1_trimmed_srt_no_dups.bw"
        peak="/media/koh/grasnas/home/data/omniplot/HepG2_KMT2B_ENCFF036WCY_srt.bed"
        peak="/media/koh/grasnas/home/data/omniplot/peaks_mm39.bed"
        #tss="/media/koh/grasnas/home/data/omniplot/gencode.v40.annotation_tss_srt.bed"
        call_superenhancer(bigwig=f, bed=peak,tss_dist=5000,plot_signals=True , gff=gff,closest_genes=True,
                           go_analysis=False,nearest_k=3)
        plt.show()
    elif test=="plot_average":
        
        peak="/media/koh/grasnas/home/data/omniplot/HepG2_KMT2B_ENCFF036WCY_srt.bed"
        files={"KMT2B":"/media/koh/grasnas/home/data/omniplot/HepG2_KMT2B-human_ENCFF709UTL.bw",
                            "KMT2A":"/media/koh/grasnas/home/data/omniplot/HepG2_KMT2A-human_ENCFF406SHU.bw"}
        plot_average(files=files,
                            bed=peak,
                            order=["KMT2B",  "KMT2A"],clustering="kmeans"
                            )
        plt.show()
    elif test=="plot_genebody":

        gff="/media/koh/grasnas/home/data/omniplot/gencode.v40.annotation.gff3"
        files={"adrenal_grand":"/media/koh/grasnas/home/data/omniplot/adrenal_grand_plus_ENCFF809PGE.bigWig"}
        files_minus={"adrenal_grand":"/media/koh/grasnas/home/data/omniplot/adrenal_grand_minus_ENCFF896WDT.bigWig"}
        _plot_genebody(files=files,files_minus=files_minus,
                            gff=gff,binsize=1,
                            order=["adrenal_grand"]#,clustering="kmeans"
                            )
        plt.show()
    #raise NotImplementedError("This function will plot ChIP-seq data.")