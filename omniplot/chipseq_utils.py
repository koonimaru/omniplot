
from typing import List,Dict,Optional,Union, Any, Iterable
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
try:
    import pyBigWig as pwg
except ImportError:
    pass
import time
import os 
import scipy.stats
from scipy.spatial.distance import squareform
from scipy.spatial import distance
from joblib import Parallel, delayed
import itertools as it
from joblib.externals.loky import get_reusable_executor
sns.set_theme(font="Arial", style={'grid.linestyle': "",'axes.facecolor': 'white'})
import itertools
from sklearn.neighbors import BallTree
from intervaltree import Interval, IntervalTree
import pickle
import os

#from cython_utils import chipseq_utils
def _range_diff(r1, r2):
    s1, e1 = r1
    s2, e2 = r2
    endpoints = sorted((s1, s2, e1, e2))
    result = []
    if endpoints[0] == s1 and endpoints[1] != s1:
        result.append((endpoints[0], endpoints[1]))
    if endpoints[3] == e1 and endpoints[2] != e1:
        result.append((endpoints[2], endpoints[3]))
    return result

def _multirange_diff(r1_list, r2_list):
    for r2 in r2_list:
        r1_list = list(itertools.chain(*[_range_diff(r1, r2) for r1 in r1_list]))
    return r1_list

class gff_parser():
    resolution=100000
    def __init__(self,gff):
        from operator import methodcaller
        time_start=time.time()
        home=os.path.expanduser('~')
        cache=os.path.join(home, "omniplot_cache")
        if not os.path.isdir(cache):
            os.makedirs(cache)
        h, t=os.path.split(gff)
        gffchache=os.path.join(cache, t+".pickle")
        gffchache_info=os.path.join(cache, t+"_info.pickle")
        if os.path.isfile(gffchache):
            with open(gffchache, "rb") as fin:
                data=pickle.load(fin)
            with open(gffchache_info, "rb") as fin:
                info=pickle.load(fin)
        else:
            data={}
            info={}
            with open(gff) as fin:
                for l in fin:
                    if l.startswith("#"):
                        if l.startswith("##gff-version"):
                            l=l.split()
                            info["gff-version"]=l[1]
                        elif l.startswith("#description:"):
                            l=l.strip("\n").split(": ")
                            info["description"]=l[1]
                        elif l.startswith("#provider:"):
                            l=l.strip("\n").split(": ")
                            info["provider"]=l[1]
                        elif l.startswith("#format"):
                            l=l.strip("\n").split(": ")
                            info["format"]=l[1]
                        elif l.startswith("#date"):
                            l=l.strip("\n").split(": ")
                            info["date"]=l[1]
                        continue
                    chrom, source, kind, s, e, _, ori, _, meta=l.split()
                    s, e=int(s)-1, int(e)
                    meta=meta.split(";")
                    tmp={}
                    for m in meta:
                        k, v=m.split("=")
                        tmp[k]=v
                    

                    # if not chrom in data:
                    #     data[chrom]={"pos":[], "kind":[],"ori":[],"meta":[]}
                    #
                    #
                    # #tmp={k: v for k,v in map(methodcaller("split", "="), meta)}
                    # data[chrom]["pos"].append([s, e])
                    # data[chrom]["kind"].append(kind)
                    # data[chrom]["ori"].append(ori)
                    # data[chrom]["meta"].append(tmp)        
                    if not chrom in data:
                        data[chrom]={}
                  
                    _s=self.resolution*(s//self.resolution)
                    if not _s in data[chrom]:
                        data[chrom][_s]={"pos":[], "kind":[],"ori":[],"meta":[]}
                    data[chrom][_s]["pos"].append([s, e])
                    data[chrom][_s]["kind"].append(kind)
                    data[chrom][_s]["ori"].append(ori)
                    data[chrom][_s]["meta"].append(tmp)  
            with open(gffchache, "wb") as fin:
                pickle.dump(data, fin)
            with open(gffchache_info, "wb") as fin:
                pickle.dump(info, fin)         
        self.data=data
        self.info=info
        print(time.time()-time_start)
    def get_genes(self, chrom: str,
                  start: int, 
                  end: int,
                  gene_type: set=set(["protein_coding"])) -> list:
        _data=self.data[chrom]
        genes=[]
        
        
        _start=start//self.resolution
        _end=end//self.resolution
        for h in range(_start, _end+1):
            _h=self.resolution*h
            _data2=_data[_h]
            for i, (se, kind, meta, ori) in enumerate(zip(_data2["pos"], _data2["kind"], _data2["meta"], _data2["ori"])):
                s, e=se
                if e < start:
                    continue
                if s>end:
                    break
                if kind=="gene" and meta["gene_type"] in gene_type:
                    
                
                    if start<=s and e<=end:
                        genename=meta["gene_name"]
                        genes.append([genename,s,e,ori])
                    elif start<=s<=end and end<e:
                        genename=meta["gene_name"]
                        genes.append([genename,s,end,ori])
                    elif s<start and start<=e<=end  and meta["gene_type"] in gene_type:
                        genename=meta["gene_name"]
                        genes.append([genename,start,e,ori])
        
        
        # _data=self.data[chrom]
        # genes=[]
        # for i, (se, kind, meta, ori) in enumerate(zip(_data["pos"], _data["kind"], _data["meta"], _data["ori"])):
        #     s, e=se
        #     if e < start:
        #         continue
        #     if s>end:
        #         break
        #     if kind=="gene" and meta["gene_type"] in gene_type:
        #
        #
        #         if start<=s and e<=end:
        #             genename=meta["gene_name"]
        #             genes.append([genename,s,e,ori])
        #         elif start<=s<=end and end<e:
        #             genename=meta["gene_name"]
        #             genes.append([genename,s,end,ori])
        #         elif s<start and start<=e<=end  and meta["gene_type"] in gene_type:
        #             genename=meta["gene_name"]
        #             genes.append([genename,start,e,ori])
                
            

        return genes
    def get_tss(self, gene_type=set(["protein_coding"])):
        
        tss={}
        
        for chrom, _data in self.data.items():
            tss[chrom]={"pos":[],"genes":[]}
            for i, (s,e) in enumerate(_data["pos"]):
                if _data["kind"][i]=="gene" and _data["meta"][i]["gene_type"] in gene_type:
                    tss[chrom]["genes"].append(_data["meta"][i]["gene_name"])
                    if _data["ori"][i]=="+":
                        tss[chrom]["pos"].append(s)
                    else:
                        tss[chrom]["pos"].append(e)
        
        return tss 
    
    def get_tss_extend(self, gene_type=set(["protein_coding"]), extend=5000):
        
        tss={}
        
        for chrom, _data in self.data.items():
            tss[chrom]={"pos":[],"genes":[]}
            for i, (s,e) in enumerate(_data["pos"]):
                if _data["kind"][i]=="gene" and _data["meta"][i]["gene_type"] in gene_type:
                    tss[chrom]["genes"].append(_data["meta"][i]["gene_name"])
                    if _data["ori"][i]=="+":
                        tss[chrom]["pos"].append([s-extend,s+extend])
                    else:
                        tss[chrom]["pos"].append([e-extend,e+extend])
        
        return tss
    
    def closest_genes(self, peaks, K=1):
        tss=self.get_tss()
        
        if type(peaks)==list:
            _peaks={}
            for chrom, s, e in peaks:
                if not chrom in _peaks:
                    _peaks[chrom]={"se":[],"center":[]}
                _peaks[chrom]["center"].append((s+e)/2)
                _peaks[chrom]["se"].append([s,e])
            peaks=_peaks
        genes_peaks=[]
        i=0
        for chrom in tss.keys():
            if not chrom in peaks:
                continue
            _tss=np.array(tss[chrom]["pos"]).reshape([-1,1])
            _genes=np.array(tss[chrom]["genes"])
            _peaks=np.array(peaks[chrom]["center"]).reshape([-1,1])
            tree = BallTree(_tss)
            dist, ind = tree.query(_peaks, k=K) 
            for _ind, _dist, (s,e) in zip(list(ind), list(dist),peaks[chrom]["se"]):
                _closest=_genes[_ind]
                genes_peaks.append([chrom, s, e, ",".join(list(_closest)), ",".join(map(str, list(_dist)))])
        return genes_peaks
    
def _calc_pearson2(_ind, _mat):
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
def _calc_pearson(_ind, _mat):
    p=scipy.stats.pearsonr(_mat[_ind[0]], _mat[_ind[1]])[0]
    #p=scipy.stats.spearmanr(_mat[_ind[0]], _mat[_ind[1]])[0]
    #p=distance.cdist([_mat[_ind[0]]], [_mat[_ind[1]]])[0][0]
    return p

def _read_bw(f, chrom, chrom_size, step):
        bw=pwg.open(f)
        val=np.array(bw.values(chrom,0,chrom_size))
        rem=chrom_size%step
        val=val[:val.shape[0]-rem]
        val=np.nan_to_num(val)
        val=val.reshape([-1,step]).mean(axis=1)
        return val
def _read_peaks(peakfile):
    peaks={}
    with open(peakfile) as fin:
        for l in fin:
            l=l.split()
            if not l[0] in peaks:
                peaks[l[0]]=[]
            peaks[l[0]].append([int(l[1]),int(l[2])])
    return peaks

def _stitching(peakfile, stitchdist):
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

def _stitching_for_pyrange(peakfile, stitchdist):
    speaks={"Chromosome":[],"Start":[],"End":[]}
    
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
        for s, e in stack:
            speaks["Chromosome"].append(chrom)
            speaks["Start"].append(s)
            speaks["End"].append(e)
        #speaks[chrom]=stack
    return speaks

def _read_tss(tss, tss_dist):
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

def _interval_subtraction(list1, list2):
    t = IntervalTree([Interval(s, e) for s, e in list1])

    newt=Parallel(n_jobs=-1)(delayed(t.chop)(s, e) for s, e in list2)
    #print(newt[:10])
    _newt=[]
    for val in newt:
        if val!=None:
            _newt.append(val)
    newt=sorted(_newt)
    return [[_t.begin, _t.end] for _t in newt]
def _remove_close_to_tss(stitched, tss_pos):
    _stitched={}
    chroms=stitched.keys()
    # tmp=[]
    # for chrom in chroms:
    #     tmp.extend(interval_subtraction(stitched[chrom], tss_pos[chrom]))
    # print(tmp[:10])
    tmp=Parallel(n_jobs=-1)(delayed(_multirange_diff)(stitched[chrom], tss_pos[chrom]) for chrom in chroms)
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

def _find_extremes(signals, pos):
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
    
    
    return x, y, pos, index, srt

def _readgff(_gff, _kind, tss_dist):
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

def _readgtf(_gff, _kind,tss_dist):
    gff_dict={"Chromosome":[], "Start": [], "End": []}
    with open(_gff) as fin:
        for l in fin:
            if l.startswith("#"):
                continue
            chrom, source, kind, s, e, _, ori, _, meta=l.strip("\n").split("\t")
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

def _readgff2(_gff, _kind, others=[]):
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
                if other in tmp:
                    gff_dict[other].append(tmp[other])
                elif other=="Strand":
                    gff_dict[other].append(ori)
    return gff_dict

def _readgff_transcripts(_gff, _extend, avoid="chrM"):
    
    transcripts={"+":{},"-":{}}
    
    with open(_gff) as fin:
        for l in fin:
            if l.startswith("#"):
                continue
            chrom, source, kind, s, e, _, ori, _, meta=l.split()
            if chrom==avoid:
                continue
            s, e=int(s)-1, int(e)
            if kind =="exon":
                tmp={}
                #print(meta)
                meta=meta.split(";")
                
                for m in meta:
                    k, v=m.split("=")
                    tmp[k]=v
                if "pseudogene" in tmp["gene_type"].split("_"):
                    continue
                tid=tmp["transcript_id"]
                if not tid in transcripts[ori]:
                    transcripts[ori][tid]=[]
                transcripts[ori][tid].append([chrom, s, e])
    _transcripts={"+":{},"-":{}}
    for strand, tids in transcripts.items():
        if strand=="+":
            for tid, exons in tids.items():
                _transcripts[strand][tid]=[]
                total_len=0
                i=0
                for chrom, s, e in exons:
            
                    total_len+=e-s
                    if i==0:
                        s=s-_extend
                        if s < 0:
                            s=0
                    if total_len>_extend:
                        e=e-(total_len-_extend)
                        if s<e:
                            _transcripts[strand][tid].append([chrom, s, e])
                        break
                    else:
                        _transcripts[strand][tid].append([chrom, s, e])
                    i+=1
                
        elif strand=="-":
            for tid, exons in tids.items():
                _transcripts[strand][tid]=[]
                total_len=0
                i=0
                for chrom, s, e in reversed(exons):
                    total_len+=e-s
                    if i==0:
                        e=e+_extend
                    if total_len>_extend:
                        s=s+(total_len-_extend)
                        if s<e:
                            _transcripts[strand][tid].append([chrom, s, e])
                        break
                    else:
                        _transcripts[strand][tid].append([chrom, s, e])
                    i+=1
    return _transcripts

def _readgtf2(_gff, _kind, others=[]):
    gff_dict={"Chromosome":[], "Start": [], "End": []}
    for other in others:
        gff_dict[other]=[]
    with open(_gff) as fin:
        for l in fin:
            if l.startswith("#"):
                continue
            chrom, source, kind, s, e, _, ori, _, meta=l.strip(";\n").split("\t")
            if kind !=_kind:
                continue
            tmp={}
            #print(meta)
            meta=meta.split("; ")
            #print(meta)
            for m in meta:
                k, v=m.split()
                v=v.strip('";')
                if k in others:
                    tmp[k]=v

            s, e=int(s)-1, int(e)
            gff_dict["Chromosome"].append(chrom)
            gff_dict["Start"].append(s)
            gff_dict["End"].append(e)
            for other in others:
                if other in tmp:
                    gff_dict[other].append(tmp[other])
                elif other=="Strand":
                    gff_dict[other].append(ori)
    return gff_dict

def _read_and_reshape_bw(chrom, s, e, _bw, binsize):
    bw=pwg.open(_bw)
    val=bw.values(chrom, s, e)
    #print(val)
    val=np.array(val)
    #print(val.shape)
    val=val.reshape([-1,binsize]).mean(axis=1)
    #print(val.shape)
    _sum=np.sum(val)
    if _sum==0:
        mean=0
    else:
        mean=_sum/(e-s)
    # mean=bw.stats(chrom, s, e, exact=True)[0]
    # if mean==None:
    #     mean=0
    return val, mean 

def _read_bw_stats(chrom, s, e, _bw):
    bw=pwg.open(_bw)
    val=bw.stats(chrom, s, e)[0]
    if val==None:
        val=0
    return val 
def _read_and_reshape_np(s, e, bw, binsize):
    val=bw[s:e]
    #print(val)
    val=np.array(val)
    #print(val.shape)
    val=val.reshape([-1,binsize]).mean(axis=1)
    #print(val.shape)
    mean=val.mean()
    if mean==None:
        mean=0
    return val, mean

def _flatten_gen_comp(lst: List[Any]) -> Iterable[Any]:
                """Flatten a list using generators comprehensions."""
                return (item
                        for sublist in lst
                        for item in sublist)
                
def _read_transcripts(_bw, _pos, binsize, extend):
    expected=2*extend//binsize
    bw=pwg.open(_bw)
    vals=[]
    for chrom, s, e in _pos:
        try:
            val=bw.values(chrom, s, e)
        except RuntimeError as e:
            print(chrom, s, e)
        vals.append(val)
        
    #print(val)
    val=np.nan_to_num(list(_flatten_gen_comp(vals)))
    
    
    
    val=val.reshape([-1,binsize]).mean(axis=1)
    if val.shape[0]<expected:
        
        val =np.concatenate([val, np.zeros([expected-val.shape[0]])])
    return val
    
    
    
    
    
    