import importlib.metadata
__version__ = importlib.metadata.version("pretty_confusion_matrix")

from .pretty_confusion_matrix import pp_matrix, pp_matrix_from_data
