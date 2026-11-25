import numpy as np

from psionic import GO_MTL, normalize_column

N_GENES, N_TFS, N_TASKS, K = 60, 8, 4, 3


def make_tasks(seed=0):
    """Small synthetic problem: tasks share two latent regulatory programs."""
    rng = np.random.default_rng(seed)
    programs = rng.normal(size=(N_TFS, 2))
    X_train, Y_train = [], []
    for _ in range(N_TASKS):
        X = normalize_column(rng.normal(size=(N_GENES, N_TFS)))   # gene x TF
        w = programs @ rng.normal(size=2)
        y = X @ w + 0.05 * rng.normal(size=N_GENES)
        X_train.append(X)
        Y_train.append(normalize_column(y[:, None]))              # gene x 1
    return X_train, Y_train


def fit(mu=0.01, seed=0):
    X_train, Y_train = make_tasks(seed)
    return GO_MTL(X_train, Y_train, k=K, Maxsteps=3, lambda_=0.01, mu=mu, O="R")


def test_output_shapes():
    L, S, W = fit()
    assert L.shape == (N_TFS, K)
    assert S.shape == (K, N_TASKS)
    assert W.shape == (N_TFS, N_TASKS)
    assert np.allclose(W, L @ S)


def test_deterministic():
    L1, S1, _ = fit()
    L2, S2, _ = fit()
    assert np.allclose(L1, L2)
    assert np.allclose(S1, S2)


def test_large_mu_gives_all_zero_S():
    _, S, _ = fit(mu=1e3)
    assert np.allclose(S, 0.0)
