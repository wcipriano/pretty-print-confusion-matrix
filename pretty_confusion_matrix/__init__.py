import importlib.metadata
try:
    __version__ = importlib.metadata.version("pretty_confusion_matrix")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

from .pretty_confusion_matrix import pp_matrix, pp_matrix_from_data
