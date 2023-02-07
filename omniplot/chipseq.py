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
from omniplot.chipseq_utils import stitching_for_pyrange, gff_parser, read_peaks, read_bw, calc_pearson,stitching,read_tss, remove_close_to_tss,find_extremes
sns.set_theme(font="Arial", style={'grid.linestyle': "",'axes.facecolor': 'whitesmoke'})
import itertools
import random
import string
from matplotlib.colors import LogNorm
from sklearn.decomposition import PCA, NMF, LatentDirichletAllocation
from scipy.stats import fisher_exact
from scipy.stats import zscore
from sklearn.cluster import KMeans
from omniplot.utils import optimal_kmeans
import time
import pyranges as pr
def plot_bigwig(files: dict, 
                bed: Union[str, list], 
                gff: str,
                step: int=100,
                stack_regions: str="horizontal", 
                highlight: Union[str, list]=[],
                highlightname=""):
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
    if type(bed)==str:
        posrange=[]
        pos=[]
        with open(bed) as fin:
            
            for l in fin:
                l=l.split()
                chrom, s, e=l[0],l[1],l[2]
                posrange.append(pr.from_dict({"Chromosome": [chrom], "Start": [int(s.replace(",",""))], "End": [int(e.replace(",",""))]}))
                pos.append([chrom,int(s.replace(",","")),int(e.replace(",",""))])
            # pos=[]
        # with open(bed) as fin:
        #     for l in fin:
        #         l=l.split()
        #         chrom, s, e=l[0],l[1],l[2]
        #         
        
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
    gff_ob=pr.read_gff3(gff)
    
    # geneset=[]
    # for chrom, start, end in pos:
    #     geneset.append(gff_ob.get_genes(chrom, start, end))
    time_start=time.time()
    geneset=[]
    for gr in posrange:
        gr2=gff_ob.intersect(gr)
        if len(gr2)==0:
            geneset.append([])
        else:
            print(gr2)
            gr2=gr2.df.loc[gr2.df["Feature"]=="gene"]
            tmp=[]
            for gene_name, start, end, ori in zip(gr2["gene_name"], gr2["Start"], gr2["End"], gr2["Strand"]):
                tmp.append([gene_name, start, end, ori])
            geneset.append(tmp)
    print(geneset)
    #geneset=Parallel(n_jobs=-1)(delayed(gff_ob.get_genes)(chrom, start, end) for chrom, start, end in pos)
    print(time.time()-time_start)
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
                print(s, e, _e, vallen, step)
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
                    axes[posi*2+1,si].plot([ge-interval/32, ge], [slot*1+0.1,slot*1], color="gray")
                elif ori=="-":
                    axes[posi*2+1,si].plot([gs, gs+interval/32], [slot*1,slot*1+0.1], color="gray")
                
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
    else:
        fig, axes=plt.subplots(nrows=len(files)+1,ncols=len(pos),gridspec_kw={
                           'height_ratios': [2]*len(files)+[1]})
        
        if len(axes.shape)==1:
            axes=axes.reshape([-1,1])
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
                    axes[si+1,posi].plot([ge-interval/32, ge], [slot*1+1,slot*1], color="gray")
                elif ori=="-":
                    axes[si+1,posi].plot([gs, gs+interval/32], [slot*1,slot*1+1], color="gray")
                
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
    return {"ax":axes,"values":mat,"genes":geneset,"positions":pos}


def plot_bigwig_correlation(files: dict, 
                            chrom: str="chr1",
                            step: int=1000,
                            palette: str="coolwarm",
                            figsize: list=[6,6],
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
    g=sns.clustermap(data=dmat,xticklabels=samples,yticklabels=samples,
               method="ward", cmap=palette,
               col_cluster=True,
               row_cluster=True,
               figsize=figsize,
               rasterized=True,cbar_kws={"label":"Pearson correlation"}, annot=annot,**clustermap_param)
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)  # For y axis
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90) # For x axis
    return {"correlation": dmat,"samples":samples}
    
def call_superenhancer(bigwig: str, 
                       peakfile: str,
                       stitch=25000,
                       tss_dist: int=0,
                       plot_signals=False,
                       gff: str="",
                       closest_genes: bool=False,
                       nearest_k: int=1,
                       go_analysis: bool=True,
                       gonames=["GO_Biological_Process_2021","Reactome_2022","WikiPathways_2019_Human"]):
    """
    Find super enhancers and plot an enhancer rank.  
    
    Parameters
    ----------
    bigwig : str
        A bigwig file.
    peakfile : str
        A peak file.
    tss : str, optional
        A bed file containing transcriptional start sites. If you want to exclude enhancers that overlap with TSSs. 
        You could create this file by scripts/gff2tss.py.
    stitch: int, optional
        The maximum distance of enhancers to stitch together.
    tss_dist : int, optional
        The distance of enhancers from TSSs to exclude.
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
    
    bw=pwg.open(bigwig)
    chrom_sizes=bw.chroms()
    peaks=read_peaks(peakfile)
    #stitched=stitching(peaks, stitch)
    
    def readgff(_gff, _kind):
        gff_dict={"Chromosome":[], "Start": [], "End": []}
        with open(_gff) as fin:
            for l in fin:
                if l.startswith("#"):
                    continue
                chrom, source, kind, s, e, _, ori, _, meta=l.split()
                if kind !=_kind:
                    continue
                s, e=int(s)-1, int(e)
                if ori=="+":
                    gff_dict["Chromosome"].append(chrom)
                    gff_dict["Start"].append(s-tss_dist)
                    gff_dict["End"].append(s+tss_dist)
                else:
                    gff_dict["Chromosome"].append(chrom)
                    gff_dict["Start"].append(e-tss_dist)
                    gff_dict["End"].append(e+tss_dist)
        return gff_dict
    def readgff2(_gff, _kind, others=[]):
        gff_dict={"Chromosome":[], "Start": [], "End": []}
        for other in others:
            gff_dict[other]=[]
        with open(_gff) as fin:
            for l in fin:
                if l.startswith("#"):
                    continue
                chrom, source, kind, s, e, _, ori, _, meta=l.split()
                if kind !=_kind:
                    continue
                tmp={}
                #print(meta)
                meta=meta.split(";")
                for m in meta:
                    k, v=m.split("=")
                    if k in others:
                        tmp[k]=v
                s, e=int(s)-1, int(e)
                gff_dict["Chromosome"].append(chrom)
                gff_dict["Start"].append(s)
                gff_dict["End"].append(e)
                for other in others:
                    gff_dict[other].append(tmp[other])
        return gff_dict
    if tss_dist != 0:
        start_time=time.time()
        stitched=pr.from_dict(stitching_for_pyrange(peaks, stitch))
        gff_dict=readgff(gff, "transcript")
        gffr=pr.from_dict(gff_dict)
        
        _stitched=stitched.subtract(gffr, nb_cpu=2)
        stitched={}
        for chrom, s, e in zip(_stitched.df["Chromosome"], _stitched.df["Start"], _stitched.df["End"]):
            if not chrom in stitched:
                stitched[chrom]=[]
            stitched[chrom].append([s,e])
        print("removing enhancers around TSS took ", time.time()-start_time)
        # tss_pos=read_tss(tss,tss_dist)
        # stitched=remove_close_to_tss(stitched, tss_pos)
    else:
        stitched=stitching(peaks, stitch)
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
    
    
    if closest_genes==True:
        gff_dict=readgff2(gff, "gene", others=["gene_name"])
        gffr=pr.from_dict(gff_dict)
        grlist=[]
        for chrom_se in pos:
            chrom, se=chrom_se.split(":")
            s, e=se.split("-")
            grlist.append(pr.from_dict({"Chromosome":[chrom],"Start":[int(s)],"End":[int(e)]}))
        _nearestgenes=Parallel(n_jobs=-1)(delayed(gr.k_nearest)(gffr, k=nearest_k) for gr in grlist)
        
    fig, ax=plt.subplots(figsize=[6,4])
    ax.scatter(x[:sindex], y[:sindex],s=5,rasterized=True)
    ax.scatter(x[sindex:], y[sindex:], color="r",s=5,rasterized=True)
    ax.plot(x,x*np.amax(y)+b, color="gray")
    
    if closest_genes==True:
        for j in range(5):
            ax.text(x[-j-1],y[-j-1], ",".join(_nearestgenes[-j-1].gene_name))
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
    plt.subplots_adjust(right=0.620, bottom=0.130)
    
    
    stitched_out=os.path.splitext(peakfile)[0]+"_SE.bed"
    with open(stitched_out, "w") as fout:
        
        if closest_genes==True:
            rank=1
            for _pos, _sig, _gr in zip(reversed(pos[sindex:]), reversed(y[sindex:]),reversed(_nearestgenes[sindex:])):
                chrom, se=_pos.split(":")
                s, e=se.split("-")
                fout.write("\t".join([chrom,str(s),str(e), ",".join(_gr.gene_name), str(_sig),str(rank)+"\n"]))
                rank+=1
        
        else:
            rank=1
            for _pos, _sig in zip(reversed(pos[sindex:]), reversed(y[sindex:])):
                chrom, se=_pos.split(":")
                s, e=se.split("-")
                fout.write("\t".join([chrom,str(s),str(e), str(_sig),str(rank)+"\n"]))
                rank+=1
    if go_analysis==True:
        try:
            import gseapy
        except ImportError:
            raise ImportError('If you want GO analysis, you need to install gseapy. Try "pip install gseapy".')
        godf={0:[],1:[]}
        y=(y-np.mean(y))/np.std(y)
        for _sig, genes in zip(reversed(y), reversed(_nearestgenes)):
            for gene in list(genes.gene_name):
                godf[0].append(gene)
                godf[1].append(_sig)
        print(godf)
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
            print(gs_res.res2d)
            for index, row in gs_res.res2d.iterrows():
                #print(row)
                if row["FDR q-val"] <0.05:
                    t=row.Term
                    gseapy.gseaplot(rank_metric=gs_res.ranking,
                             term=t,
                             cmap=plt.cm.seismic,
                             **gs_res.results[t],
                             ofname=os.path.splitext(peakfile)[0]+"_"+name+"_gsea.pdf"
                             )
    
    if plot_signals:
        if gff=="":
            raise Exception("Please provide a gff file.")
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
                    
        plot_bigwig(files={"Sample": bigwig}, 
                    bed=tmpfile, gff=gff, 
                    step=100,
                    stack_regions="vertical",
                    highlight=highlights)
        os.remove(tmpfile)
def plot_average(files: dict, 
                 bed: Union[str,list], 
                 order: list=[], extend: int=500, 
                 palette: str="coolwarm",
                 binsize: int=10,
                 clustering: str="",
                 n_clusters: int=5,
                 orientaion=False):
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
    if type(bed)==str:
        
        pos=[]
        with open(bed) as fin:
            for l in fin:
                l=l.split()
                chrom, s, e=l[0],l[1],l[2]
                s, e=int(s.replace(",","")),int(e.replace(",",""))
                center=(s+e)//2
                pos.append([chrom,center-extend,center+extend])
    else:
        pos=bed
    
    if len(order)==0:
        order=list(files.keys())
    data={}
    data_mean=[]
    
    for i, sample in enumerate(order):
        bigwig=files[sample]
        bw=pwg.open(bigwig)
        data[sample]=[]
        
        for chrom, s, e in pos:
            val=bw.values(chrom, s, e)
            #print(val)
            val=np.array(val)
            #print(val.shape)
            val=val.reshape([-1,binsize]).mean(axis=1)
            #print(val.shape)
            data[sample].append(val)
            if i==0:
                data_mean.append(bw.stats(chrom, s, e, exact=True)[0])
    labels=[]
    if clustering=="kmeans_all":
        mat=[]
        for sample in order:
            print(np.array(data[sample]).shape)
            mat.append(data[sample])
        mat=np.concatenate(mat, axis=1)
        print(mat.shape)
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
        optimalclusternum=optimal_kmeans(mat, [2, 10])
        n_clusters=np.amax(optimalclusternum)
        kmean = KMeans(n_clusters=n_clusters, random_state=0,n_init=10)
        kmX=kmean.fit(mat)
        labels=kmX.labels_
        
    sortindex=np.argsort(data_mean)[::-1]
    pos=np.array(pos)[sortindex]

    plotindex=np.arange(0, len(pos), len(pos)//1000)
    fig, axes=plt.subplots(ncols=len(order), figsize=[2*len(order), 6])
    for i, (sample, ax) in enumerate(zip(order, axes)):
        vals=data[sample]

        vals=np.array(vals)

        vals=vals[sortindex]
        if clustering!="":
            
            _labels=np.array(labels)[sortindex]
            sortindex2=np.argsort(_labels)
            vals=vals[sortindex2]
            pos=[sortindex2]
            _labels=_labels[sortindex2]
            _labels=_labels[plotindex]
            ulabel, clabel=np.unique(_labels, return_counts=True)

        vals=vals[plotindex]
        im=ax.imshow(vals,aspect="auto", cmap=palette, interpolation="none",norm=LogNorm(vmin=np.quantile(vals,0.05)))
        
        
        
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
    fig, axes=plt.subplots(ncols=len(order), figsize=[4*len(order), 3])
    maxval=0
    for i, (sample, ax) in enumerate(zip(order, axes)):
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
    
    return {"values":data,"potisons":pos,"labels":labels}



def plot_genebody(files: dict, 
                 bed: Union[str,list], 
                 order: list=[], extend: int=500, 
                 palette: str="coolwarm",
                 binsize: int=10,
                 clustering: str="",
                 n_clusters: int=5):
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
    if type(bed)==str:
        
        pos=[]
        with open(bed) as fin:
            for l in fin:
                l=l.split()
                chrom, s, e=l[0],l[1],l[2]
                s, e=int(s.replace(",","")),int(e.replace(",",""))
                center=(s+e)//2
                pos.append([chrom,center-extend,center+extend])
    else:
        pos=bed
    
    if len(order)==0:
        order=list(files.keys())
    data={}
    data_mean=[]
    
    for i, sample in enumerate(order):
        bigwig=files[sample]
        bw=pwg.open(bigwig)
        data[sample]=[]
        
        for chrom, s, e in pos:
            val=bw.values(chrom, s, e)
            #print(val)
            val=np.array(val)
            #print(val.shape)
            val=val.reshape([-1,binsize]).mean(axis=1)
            #print(val.shape)
            data[sample].append(val)
            if i==0:
                data_mean.append(bw.stats(chrom, s, e, exact=True)[0])
    labels=[]
    if clustering=="kmeans_all":
        mat=[]
        for sample in order:
            print(np.array(data[sample]).shape)
            mat.append(data[sample])
        mat=np.concatenate(mat, axis=1)
        print(mat.shape)
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
        optimalclusternum=optimal_kmeans(mat, [2, 10])
        n_clusters=np.amax(optimalclusternum)
        kmean = KMeans(n_clusters=n_clusters, random_state=0,n_init=10)
        kmX=kmean.fit(mat)
        labels=kmX.labels_
        
    sortindex=np.argsort(data_mean)[::-1]
    pos=np.array(pos)[sortindex]

    plotindex=np.arange(0, len(pos), len(pos)//1000)
    fig, axes=plt.subplots(ncols=len(order), figsize=[2*len(order), 6])
    for i, (sample, ax) in enumerate(zip(order, axes)):
        vals=data[sample]

        vals=np.array(vals)

        vals=vals[sortindex]
        if clustering!="":
            
            _labels=np.array(labels)[sortindex]
            sortindex2=np.argsort(_labels)
            vals=vals[sortindex2]
            pos=[sortindex2]
            _labels=_labels[sortindex2]
            _labels=_labels[plotindex]
            ulabel, clabel=np.unique(_labels, return_counts=True)

        vals=vals[plotindex]
        im=ax.imshow(vals,aspect="auto", cmap=palette, interpolation="none",norm=LogNorm(vmin=np.quantile(vals,0.05)))
        
        
        
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
    fig, axes=plt.subplots(ncols=len(order), figsize=[4*len(order), 3])
    maxval=0
    for i, (sample, ax) in enumerate(zip(order, axes)):
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
    
    return {"values":data,"potisons":pos,"labels":labels}

if __name__=="__main__":
    #test="plot_bigwig"
    test="plot_bigwig_correlation"
    
    test="plot_average"
    test="plot_bigwig"
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
        gff="/media/koh/grasnas/home/data/omniplot/gencode.v40.annotation.gff3"
        f="/media/koh/grasnas/home/data/omniplot/HepG2_KMT2B-human_ENCFF709UTL.bw"
        peak="/media/koh/grasnas/home/data/omniplot/HepG2_KMT2B_ENCFF036WCY_srt.bed"
        tss="/media/koh/grasnas/home/data/omniplot/gencode.v40.annotation_tss_srt.bed"
        call_superenhancer(bigwig=f, peakfile=peak,tss_dist=5000,plot_signals=True , gff=gff,closest_genes=True,go_analysis=True)
        plt.show()
    elif test=="plot_average":
        
        peak="/media/koh/grasnas/home/data/omniplot/HepG2_KMT2B_ENCFF036WCY_srt.bed"
        tss="/media/koh/grasnas/home/data/omniplot/gencode.v40.annotation_tss_srt.bed"
        files={"KMT2B":"/media/koh/grasnas/home/data/omniplot/HepG2_KMT2B-human_ENCFF709UTL.bw",
                            "KMT2A":"/media/koh/grasnas/home/data/omniplot/HepG2_KMT2A-human_ENCFF406SHU.bw"}
        plot_average(files=files,
                            bed=peak,
                            order=["KMT2B",  "KMT2A"],clustering="kmeans"
                            )
        plt.show()
    #raise NotImplementedError("This function will plot ChIP-seq data.")