![Downloads](https://img.shields.io/pypi/dm/omniplot)
[![PyPI Version](https://img.shields.io/pypi/v/omniplot)](https://pypi.org/project/omniplot/)
![License](https://img.shields.io/pypi/l/omniplot)
![omniplot logo][logo-image]

## What is omniplot

omniplot is a python module to draw a scientific plot with hassle free. It mainly focuses on bioinfomatics data.

<img src="images/example13.png" width="340"/> <img src="images/example2.png" width="190"/> <img src="images/example3.png" width="270"/> <br>
<img src="images/example12.png" width="200"/> <img src="images/example10.png" width="300"/> <img src="images/example15.png" width="210"/><br>
<img src="images/example5.png" width="400"/> <img src="images/example6.png" width="260"/> <br>
<img src="images/example11.png" width="600"/> <br>
## Motivation
Although there exist many good python data visualization libraries, such as 
[matplotlib](https://matplotlib.org/), 
[pandas](https://pandas.pydata.org/), 
[seaborn](https://seaborn.pydata.org/), 
[plotly](https://plotly.com/), 
[vedo](https://vedo.embl.es/) and so on,
still several kinds of plots cannot be drawn without hassle. This module is aimed to provide convenient 
tools that allow users to draw complex plots, such as a scatter plot with PCA and loadings or clustering analysis in one liner.

## Install
omniplot best works with python3.8. But, greater python versions may be OK. Please try installation with conda, if something wrong with pip installation.

```bash
pip install cython --user
git clone https://github.com/koonimaru/omniplot.git --user
cd omniplot
pip install .
```
or 

```bash
pip install cython --user
pip install git+https://github.com/koonimaru/omniplot.git --user
```
or

```bash
git clone https://github.com/koonimaru/omniplot.git
cd omniplot
conda env create -f environment.yml python=3.8
conda activate omniplot
conda install ipykernel
ipython kernel install --user --name=omniplot
conda deactivate

```
And [this](https://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-different-environments) is how to use conda environment in jupyerlab.

Known issues:<br>
If you get errors saying "error: invalid command 'bdist_wheel'", please try pip install --upgrade pip wheel setuptools

## How to use
I created jupyter notebooks to demonstrate the usage of omniplot [Link](https://github.com/koonimaru/omniplot/tree/main/ipynb).
You can open jupyter notebooks with [jupyter lab](https://jupyterlab.readthedocs.io/en/stable/) or [VScode](https://code.visualstudio.com/).

And you may want to visit an auto-generated [API](https://koonimaru.github.io/omniplot/). 

## Example usage
```python
import pandas as pd
from omniplot import plot as op
import seaborn as  sns
import matplotlib.pyplot as plt

df=sns.load_dataset("titanic")
df=df[["class","embark_town","sex"]].fillna("NA")
op.nested_piechart(df, category=["class","embark_town","sex"], title="Titanic", ignore=0.01, show_legend=True,show_values=False,hatch=True,ncols=3)
plt.show()

```
## Example usage
```python
import seaborn as sns
from omniplot import plot as op
import matplotlib.pyplot as plt
df=sns.load_dataset("penguins")
df=df.dropna(axis=0)
res=op.radialtree(df, category=["species","island","sex"])
plt.show()
```


[logo-image]: images/logo.png
