
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