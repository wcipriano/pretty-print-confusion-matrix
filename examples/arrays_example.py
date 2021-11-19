import numpy as np

from pretty_confusion_matrix import pp_matrix_from_data

y_test = np.array([1, 2, 3, 4, 5])
predic = np.array([3, 2, 4, 3, 5])

cmap = "PuRd"
pp_matrix_from_data(y_test, predic, cmap=cmap)
