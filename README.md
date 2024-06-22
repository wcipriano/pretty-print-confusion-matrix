![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pretty-confusion-matrix?logo=python&logoColor=%23FFFFFF)
<a href="https://pypi.org/project/pretty-confusion-matrix/"><img alt="PyPI" src="https://img.shields.io/pypi/v/pretty-confusion-matrix?logo=pypi&logoColor=%23FFFFFF"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg?logo=codeclimate&logoColor=%23FFFFFF"></a>
![PyPI - Wheel](https://img.shields.io/pypi/wheel/pretty-confusion-matrix)
<a href="https://libraries.io/pypi/pretty-confusion-matrix"><img alt="GitHub Repo stars" src="https://img.shields.io/librariesio/github/wcipriano/pretty-print-confusion-matrix"></a>
<a href="https://github.com/wcipriano"><img alt="GitHub Repo stars" src="https://img.shields.io/github/license/wcipriano/pretty-print-confusion-matrix?logo=apache"></a>
<a href="https://github.com/wcipriano/pretty-print-confusion-matrix/blob/master/LICENSE"><img alt="GitHub License" src="https://img.shields.io/github/stars/wcipriano/pretty-print-confusion-matrix?style=flat&logo=github"></a>
![PyPI - Downloads](https://img.shields.io/pypi/dm/pretty-confusion-matrix?logo=download)

# Confusion Matrix in Python
Plot a pretty confusion matrix (like Matlab) in python using seaborn and matplotlib

Created on Mon Jun 25 14:17:37 2018
@author: Wagner Cipriano - wagnerbhbr


This module get a pretty print confusion matrix from a NumPy matrix or from 2 NumPy arrays (`y_test` and `predictions`).

## Installation
```bash
pip install pretty-confusion-matrix
```

## Get Started

### Plotting from DataFrame:
```python
import numpy as np
import pandas as pd
from pretty_confusion_matrix import pp_matrix

array = np.array([[13,  0,  1,  0,  2,  0],
                  [0, 50,  2,  0, 10,  0],
                  [0, 13, 16,  0,  0,  3],
                  [0,  0,  0, 13,  1,  0],
                  [0, 40,  0,  1, 15,  0],
                  [0,  0,  0,  0,  0, 20]])

# get pandas dataframe
df_cm = pd.DataFrame(array, index=range(1, 7), columns=range(1, 7))
# colormap: see this and choose your more dear
cmap = 'PuRd'
pp_matrix(df_cm, cmap=cmap)
```
![alt text](https://raw.githubusercontent.com/khuyentran1401/pretty-print-confusion-matrix/master/Screenshots/Conf_matrix_default.png)


### Plotting from vectors


```python
import numpy as np
from pretty_confusion_matrix import pp_matrix_from_data

y_test = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2,
                  3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
predic = np.array([1, 2, 4, 3, 5, 1, 2, 4, 3, 5, 1, 2, 3, 4, 4, 1, 4, 3, 4, 5, 1, 2, 4, 4, 5, 1, 2, 4, 4, 5, 1, 2, 4, 4, 5, 1, 2, 4, 4, 5, 1, 2, 3, 3, 5, 1, 2, 3, 3, 5, 1, 2,
                  3, 4, 4, 1, 2, 3, 4, 1, 1, 2, 3, 4, 1, 1, 2, 3, 4, 1, 1, 2, 4, 4, 5, 1, 2, 4, 4, 5, 1, 2, 4, 4, 5, 1, 2, 4, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5])

pp_matrix_from_data(y_test, predic)
```

![alt text](https://raw.githubusercontent.com/khuyentran1401/pretty-print-confusion-matrix/master/Screenshots/Conf_matrix_default_2.png)


## Using custom labels in axis
You can customize the labels in axis, whether by DataFrame or vectors.

### From DataFrame
To plot the matrix with text labels in axis rather than integer, change the params `index` and `columns` of your dataframe.
Getting the example one above, just change the line `df_cm = pd.DataFrame(array, index=range(1, 7), columns=range(1, 7))` by
```python
col = ['Dog', 'Cat', 'Mouse', 'Fox', 'Bird', 'Chicken']
df_cm = pd.DataFrame(array, index=col, columns=col)
```
It'll replace the integer labels (**1...6**) in the axis, by **Dog, Cat, Mouse**, and so on..


### From vectors
It's very similar, in this case you just need to use the `columns` param like the example below.
This param is a positional array, i.e., the order must be the same of the data representation. 
In this example _Dog_ will be assigned to the class 0, _Cat_ will be assigned to the class 1, and so on and so forth.
Getting the example two above, just change the line `pp_matrix_from_data(y_test, predic)`, by
```python
columns = ['Dog', 'Cat', 'Mouse', 'Fox', 'Bird'] 
pp_matrix_from_data(y_test, predic, columns)
```
It'll replace "class A, ..., class E" in the axis, by **Dog, Cat, ..., Bird**.

More information about "_How to plot confusion matrix with string axis rather than integer in python_" in [this Stackoverflow answer](https://stackoverflow.com/a/51176855/1809554).



## Choosing Colormaps

You can choose the layout of the your matrix by a lot of colors options like PuRd, Oranges and more... 
To customizer your color scheme, use the param cmap of funcion pp_matrix. 
To see all the colormap available, please do this:
```python
from matplotlib import colormaps
list(colormaps)
```

More information about Choosing Colormaps in Matplotlib is available [here](https://matplotlib.org/stable/users/explain/colors/colormaps.html).




## References:
### 1. MATLAB confusion matrix:

a) [Plot Confusion](https://www.mathworks.com/help/nnet/ref/plotconfusion.html)
   
b) [Plot Confusion Matrix Using Categorical Labels](https://www.mathworks.com/help/examples/nnet/win64/PlotConfusionMatrixUsingCategoricalLabelsExample_02.png)



### 2. Examples and more on Python:

  a) [How to plot confusion matrix with string axis rather than integer in python](https://stackoverflow.com/questions/5821125/how-to-plot-confusion-matrix-with-string-axis-rather-than-integer-in-python/51176855#51176855)
  
  b) [Plot-scikit-learn-classification-report](https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report)
  
  c) [Plot-confusion-matrix-with-string-axis-rather-than-integer-in-Python](https://stackoverflow.com/questions/5821125/how-to-plot-confusion-matrix-with-string-axis-rather-than-integer-in-python)
  
  d) [Seaborn heatmap](https://www.programcreek.com/python/example/96197/seaborn.heatmap)
  
  e) [Sklearn-plot-confusion-matrix-with-labels](https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels/31720054)

  f) [Model-selection-plot-confusion-matrix](http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py)

