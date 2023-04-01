
import sys, os 
import sys
#sys.path.append("../omniplot")
from omniplot  import chipseq as ochip
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 


if __name__=="__main__":
    #test="plot_bigwig"
    test="plot_bigwig_correlation"
    
    #test="plot_bigwig"
    
    
    test="plot_average"
    #test="plot_genebody"
    #test="call_superenhancer"
    #test="plot_genebody"
    import glob
    test="plot_bed_correlation"
    if test=="plot_bed_correlation":
        fs= {"KMT2A":"/media/koh/grasnas/home/data/omniplot/KMT2A_ENCFF644EPI_srt.bed",
             "KMT2B":"/media/koh/grasnas/home/data/omniplot/HepG2_KMT2B_ENCFF036WCY_srt.bed",
             "HNF4G_rep1":"/media/koh/grasnas/home/data/omniplot/HNF4G_ENCFF413UQM_srt.bed",
             "HNF4G_rep2":"/media/koh/grasnas/home/data/omniplot/HNF4G_ENCFF123FQW_srt.bed",
             "HNF1A_rep1":"/media/koh/grasnas/home/data/omniplot/HNF1A_ENCFF696TGC_srt.bed",
             "HNF1A_rep2":"/media/koh/grasnas/home/data/omniplot/HNF1A_ENCFF162SGM_srt.bed",}
        
        ochip.plot_bed_correlation(fs,step=1000, method="pearson")
        plt.show()
    elif test=="plot_bigwig":
        fs= {"KMT2A":"/media/koh/grasnas/home/data/omniplot/HepG2_KMT2A-human_ENCFF406SHU.bw",
             "KMT2B":"/media/koh/grasnas/home/data/omniplot/HepG2_KMT2B-human_ENCFF709UTL.bw"}
        bed="/media/koh/grasnas/home/data/omniplot/tmp3.bed"
        gff="/media/koh/grasnas/home/data/omniplot/gencode.v40.annotation.gff3"
        
        ochip.plot_bigwig(fs,bed,gff, step=10)
        plt.show()
    elif test=="plot_bigwig_correlation":
        fs= {"KMT2A":"/media/koh/grasnas/home/data/omniplot/HepG2_KMT2A-human_ENCFF406SHU.bw",
             "KMT2B":"/media/koh/grasnas/home/data/omniplot/HepG2_KMT2B-human_ENCFF709UTL.bw",
             "HNF1A_rep1":"/media/koh/grasnas/home/data/omniplot/HepG2_HNF1A-human_ENCFF397BTX.bw",
             "HNF1A_rep2":"/media/koh/grasnas/home/data/omniplot/HepG2_HNF1A-human_ENCFF502ACF.bw",
             "HNF4A_rep1":"/media/koh/grasnas/home/data/omniplot/HepG2_HNF4A-human_ENCFF467SMI.bw",
             "HNF4A_rep2":"/media/koh/grasnas/home/data/omniplot/HepG2_HNF4A-human_ENCFF712VYA.bw",}
        
        ochip.plot_bigwig_correlation(fs,step=1000)
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
        ochip.call_superenhancer(bigwig=f, bed=peak,tss_dist=5000,plot_signals=True , gff=gff,closest_genes=True,
                           go_analysis=False,nearest_k=3)
        plt.show()
    elif test=="plot_average":
        
        peak="/media/koh/grasnas/home/data/omniplot/HepG2_KMT2B_ENCFF036WCY_srt.bed"
        files={"KMT2B":"/media/koh/grasnas/home/data/omniplot/HepG2_KMT2B-human_ENCFF709UTL.bw",
                            "KMT2A":"/media/koh/grasnas/home/data/omniplot/HepG2_KMT2A-human_ENCFF406SHU.bw"}
        ochip.plot_average(files=files,
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