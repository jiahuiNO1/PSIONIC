import numpy as np

def normalize_column(x):
    """
    Normalize each column of a 2D matrix to unit L2 length.
    
    Parameters
    ----------
    x : numpy.ndarray
        2D array (n_samples x n_features)

    Returns
    -------
    y : numpy.ndarray
        Column-normalized array where each column has L2 norm = 1
    """
    if x.ndim != 2:
        raise ValueError("Input must be a 2D matrix")

    # Compute L2 norm of each column
    col_norms = np.linalg.norm(x, axis=0)

    # Prevent division by zero (replace 0 norms with 1)
    col_norms_safe = np.where(col_norms == 0, 1, col_norms)

    # Normalize each column
    y = x / col_norms_safe

    return y
