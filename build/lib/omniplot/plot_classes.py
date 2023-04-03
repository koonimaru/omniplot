import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Set, Union, Any
from matplotlib.backends.backend_pdf import PdfPages
import os
import numpy as np
import pandas as pd
class _Basic_plot():
    
    def __init__(self,
                df: pd.DataFrame,
                x: str="",
                y: str="",
                variables: list=[],
                category: Union[str, list]="",
                ax: Optional[plt.Axes]=None,
                ncols: int=1,
                nrows: int=1,
                figsize: List[float]=[5,5],  
                save: str="",
                palette: str="",
                palette_list=[],
                fonts: dict={},
                show_legend: bool=True,
                dpi: float=400,
                xformat: str="",
                yformat: str="",
                xunit: str="",
                yunit:str="",
                title: str="",
                logscalex: bool=False,
                logscaley: bool=False,
                xlabel: bool=False,
                ylabel: bool=False
                ):
        
        self.save=save
        self.xscale=logscalex
        self.yscale=logscaley
        self.xformat=xformat
        self.yformat=yformat
        self.xunit=xunit
        self.yunit=yunit
        self.title=title
        self.palette=palette
        self.fonts=fonts
        self.show_legend=show_legend
        self.dpi=dpi
        if axes==None:
            fig, axes=plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
            self.fig=fig 
            self.axes=axes.flatten()
        else:
            if type(ax)==plt.Axes:
                axes=np.array([axes])
            self.axes=axes.flatten()
            self.fig=None

    def set_axis_format(self):
        ...
    def plot(self):
        ...

    def process_data(self):
        ...

    def _save(self, suffix):
        save=self.save
        if save !="":
            if save.endswith(".pdf") or save.endswith(".png") or save.endswith(".svg"):
                h, t=os.path.splitext(save)
                plt.savefig(h+"_"+suffix+t, dpi=self.dpi)
            else:
                plt.savefig(save+"_"+suffix+".pdf", dpi=self.dpi)

class _Multiple_variables(_Basic_plot):
    
    def __init__(self):
        super().__init__()
    
    
    def _separate_data(self, df, variables=[], 
                   category=""):
        if len(variables) !=0:
            X = df[variables].values
            if len(category) !=0:
                if type(category)==str:
                    category=[category]
                #category_val=df[category].values
            
            if X.dtype!=float: 
                raise TypeError(f"variables must contain only float values.")
        elif len(category) !=0:
            if type(category)==str:
                category=[category]
            #category_val=df[category].values
            df=df.drop(category, axis=1)
            X = df.values
            if X.dtype!=float: 
                raise TypeError(f"data must contain only float values except {category} column. \
            or you can specify the numeric variables with the option 'variables'.")
            
        else:    
            X = df.values
            #category_val=[]
            if X.dtype!=float: 
                raise TypeError(f"data must contain only float values. \
            or you can specify the numeric variables with the option 'variables'.")
            
        return X, category
        
        