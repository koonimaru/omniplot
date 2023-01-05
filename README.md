# omniplot

## What is omniplot

omniplot is a python module to draw a scientific plot with hassle free.
![example](example.png "example")

## Install

git clone https://github.com/koonimaru/omniplot.git <br>
cd omniplot <br>
pip install .

## Example usage
```python
import scipy.cluster.hierarchy as sch
import numpy as np
from omniplot import omniplot as op

np.random.seed(1)
numleaf=100
_alphabets=[chr(i) for i in range(97, 97+24)]
labels=sorted(["".join(list(np.random.choice(_alphabets, 10))) for i in range(numleaf)])
x = np.random.rand(numleaf)
D = np.zeros([numleaf,numleaf])
for i in range(numleaf):
    for j in range(numleaf):
        D[i,j] = abs(x[i] - x[j])
Y = sch.linkage(D, method='single')
Z2 = sch.dendrogram(Y,labels=labels,no_plot=True)
type_num=6
type_list=["ex"+str(i) for i in range(type_num)]
sample_classes={"example_color": [np.random.choice(type_list) for i in range(numleaf)]}
op.radialtree(Z2, sample_classes=sample_classes)
```