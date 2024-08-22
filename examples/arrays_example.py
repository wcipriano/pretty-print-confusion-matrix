import numpy as np

from pretty_confusion_matrix import pp_matrix_from_data

y_test = np.array([1, 2, 3, 4, 5])
predic = np.array([3, 2, 4, 3, 5])

cmap = "PuRd"
path_to_save_img = './matrix-output-from-array.png'
title = 'Confusion matrix from array'
pp_matrix_from_data(y_test, predic, cmap=cmap, path_to_save_img=path_to_save_img, title=title)
