import random
import string
from pathlib import Path
import numpy as np
import pandas as pd
import os
from pretty_confusion_matrix import pp_matrix_from_data, pp_matrix


def generate_random_string(length):
    # characters = string.ascii_letters + string.digits  # a-z, A-Z, 0-9
    characters = string.ascii_lowercase + string.digits  # a-z, 0-9
    return "".join(random.choices(characters, k=length))


def get_path(name, randon_len=8):
    rs = generate_random_string(randon_len)
    return f"./{name}-{rs}.png"


def test_array():
    """
    test ppcm with arrays:
      * true values (y)
      * predicted values (predic)
    """
    path_to_save_img = get_path("output-from-array")

    y_test = np.array([1, 2, 3, 4, 5])
    predic = np.array([3, 2, 4, 3, 5])
    cmap = "PuRd"
    title = "Confusion matrix from array"
    pp_matrix_from_data(
        y_test,
        predic,
        cmap=cmap,
        path_to_save_img=path_to_save_img,
        title=title,
    )

    file_path = Path(path_to_save_img)
    assert file_path.is_file() is True
    os.remove(path_to_save_img)


def test_df():
    """
    test ppcm with dataframe
    """
    path_to_save_img = get_path("output-from-df")
    # get pandas dataframe
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
    df_cm = pd.DataFrame(array, index=range(1, 7), columns=range(1, 7))
    # colormap: see this and choose your more dear
    cmap = "PuRd"
    title = "Confusion matrix from dataframe"
    pp_matrix(df_cm, cmap=cmap, path_to_save_img=path_to_save_img, title=title)

    file_path = Path(path_to_save_img)
    assert file_path.is_file() is True
    os.remove(path_to_save_img)
