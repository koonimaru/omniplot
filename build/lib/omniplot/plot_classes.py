import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Set, Union, Any
from matplotlib.backends.backend_pdf import PdfPages
import os
class _Basic_plot():
    
    def __init__(self, 
                 ax: Optional[plt.Axes]=None,
                 ncols: int=1,
                 nrows: int=1,
                 figsize: List[float]=[5,5],  
                 save: str="",
                 yscale: str="",
                 xscale: str="",
                 palette: str="",
                 fonts: dict={},
                 show_legend: bool=True,
                 dpi: float=400, title=""):
        
        self.save=save
        self.xscale=xscale
        self.yscale=yscale
        self.palette=palette
        self.fonts=fonts
        self.show_legend=show_legend
        self.dpi=dpi
        if ax==None:
            fig, ax=plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
            self.fig=fig 
            self.ax=ax.flatten()
        else:
            self.ax=ax
            self.fig=None
        
        
        
    
    

    def _multipage(self, filename, figs=None, dpi=400):
        pp = PdfPages(filename)
        if figs is None:
            figs = [plt.figure(n) for n in plt.get_fignums()]
        for fig in figs:
            fig.savefig(pp, format='pdf')
        pp.close()
        
    def _save(self, suffix):
        save=self.save
        if save !="":
            if save.endswith(".pdf") or save.endswith(".png") or save.endswith(".svg"):
                h, t=os.path.splitext(save)
                plt.savefig(h+"_"+suffix+t)
            else:
                plt.savefig(save+"_"+suffix+".pdf")    