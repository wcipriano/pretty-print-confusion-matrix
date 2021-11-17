import numpy as np
import pandas as pd

from pretty_confusion_matrix import pp_matrix

array = np.array(
    [
        [13, 0, 1, 0, 2, 0],
        [0, 50, 2, 0, 10, 0],
        [0, 13, 16, 0, 0, 3],
        [0, 0, 0, 13, 1, 0],
        [0, 40, 0, 1, 15, 0],
        [0, 0, 0, 0, 0, 20],
    ]
)

# get pandas dataframe
df_cm = pd.DataFrame(array, index=range(1, 7), columns=range(1, 7))
# colormap: see this and choose your more dear
cmap = "PuRd"
pp_matrix(df_cm, cmap=cmap)
