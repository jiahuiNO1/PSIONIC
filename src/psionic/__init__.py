"""
PSIONIC: chromatin-informed multitask learning of cell-type-specific
transcription factor (TF) regulatory programs.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("psionic")
except PackageNotFoundError:  # running from a source checkout without installing
    __version__ = "0.1.0"

from .tf_target import create_TF_target_matrix, combine_tf_results
from .tf_target_helpers import set_log_file
from .gomtl import GO_MTL
from .preprocessing import normalize_column

__all__ = [
    "__version__",
    "create_TF_target_matrix",
    "combine_tf_results",
    "GO_MTL",
    "normalize_column",
    "set_log_file",
]
