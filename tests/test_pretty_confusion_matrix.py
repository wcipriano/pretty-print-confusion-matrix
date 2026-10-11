import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from pretty_confusion_matrix import pp_matrix, pp_matrix_from_data

@pytest.fixture
def binary_data():
    y_test = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    predictions = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    return y_test, predictions

@pytest.fixture
def multiclass_data():
    y_test = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
    predictions = np.array([1, 2, 2, 1, 2, 3, 3, 2, 3])
    return y_test, predictions

@pytest.fixture
def zero_counts_df():
    array = np.array([[0, 0, 0], [0, 5, 0], [0, 0, 0]])
    return pd.DataFrame(array, index=["A", "B", "C"], columns=["A", "B", "C"])


def test_pp_matrix_from_data_binary(binary_data):
    y_test, predictions = binary_data
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        pass
    try:
        pp_matrix_from_data(y_test, predictions, path_to_save_img=tmp.name)
        assert os.path.exists(tmp.name)
        assert os.path.getsize(tmp.name) > 0
    finally:
        os.remove(tmp.name)

def test_pp_matrix_from_data_multiclass(multiclass_data):
    y_test, predictions = multiclass_data
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        pass
    try:
        pp_matrix_from_data(y_test, predictions, path_to_save_img=tmp.name, pred_val_axis='x')
        assert os.path.exists(tmp.name)
        assert os.path.getsize(tmp.name) > 0
    finally:
        os.remove(tmp.name)

def test_pp_matrix_zero_counts(zero_counts_df):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        pass
    try:
        pp_matrix(zero_counts_df, path_to_save_img=tmp.name)
        assert os.path.exists(tmp.name)
        assert os.path.getsize(tmp.name) > 0
    finally:
        os.remove(tmp.name)

def test_show_null_values_0(zero_counts_df):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        pass
    try:
        pp_matrix(zero_counts_df, show_null_values=0, path_to_save_img=tmp.name)
        assert os.path.exists(tmp.name)
        assert os.path.getsize(tmp.name) > 0
    finally:
        os.remove(tmp.name)

def test_show_null_values_1(zero_counts_df):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        pass
    try:
        pp_matrix(zero_counts_df, show_null_values=1, path_to_save_img=tmp.name)
        assert os.path.exists(tmp.name)
        assert os.path.getsize(tmp.name) > 0
    finally:
        os.remove(tmp.name)

def test_show_null_values_2(zero_counts_df):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        pass
    try:
        pp_matrix(zero_counts_df, show_null_values=2, path_to_save_img=tmp.name)
        assert os.path.exists(tmp.name)
        assert os.path.getsize(tmp.name) > 0
    finally:
        os.remove(tmp.name)

def test_empty_counts_df():
    array = np.array([[0, 0], [0, 0]])
    df = pd.DataFrame(array, index=["A", "B"], columns=["A", "B"])
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        pass
    try:
        pp_matrix(df, path_to_save_img=tmp.name)
        assert os.path.exists(tmp.name)
        assert os.path.getsize(tmp.name) > 0
    finally:
        os.remove(tmp.name)

def test_pp_matrix_from_data_custom_columns(binary_data):
    y_test, predictions = binary_data
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        pass
    try:
        pp_matrix_from_data(y_test, predictions, columns=["Negative", "Positive"], path_to_save_img=tmp.name)
        assert os.path.exists(tmp.name)
        assert os.path.getsize(tmp.name) > 0
    finally:
        os.remove(tmp.name)
