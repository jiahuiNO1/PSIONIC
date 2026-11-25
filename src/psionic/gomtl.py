import numpy as np
from numpy.linalg import svd, solve

from . import gomtl_helpers as mainfun



# -------------------------------------------------------------
# Loss functions (Squared Error and Logistic Loss)
# -------------------------------------------------------------
def squared_error(w, X, y):
    w = w.ravel()
    y = y.ravel()
    residual = X @ w - y
    f = np.sum(residual ** 2)
    g = 2 * X.T @ residual
    H = 2 * X.T @ X
    return f, g.reshape(-1, 1), H

def logistic_loss(w, X, y):
    w = w.ravel()
    y = y.ravel()
    Xw  = X @ w
    yXw = y * Xw
    f   = np.sum(np.logaddexp(0, -yXw))
    sig = 1.0 / (1.0 + np.exp(-yXw))
    g   = -(X.T @ (y * (1 - sig)))
    H   = X.T @ (X * (sig * (1 - sig))[:, None])
    return f, g.reshape(-1, 1), H


# -------------------------------------------------------------
# Penalized L2 objective
# -------------------------------------------------------------
def penalized_l2(w, grad_func, lambda_vec, *args):
    # Match MATLAB: if scalar, expand to vector
    if np.isscalar(lambda_vec):
        lambda_vec = lambda_vec * np.ones(len(w))

    f, g, H = grad_func(w, *args)

    g = g.ravel() 

    # Add L2 penalty to f
    f = f + np.sum(lambda_vec * (w ** 2))

    # Add L2 penalty to g
    g = g + 2 * lambda_vec * w

    # Add L2 penalty to H
    H = H + np.diag(2 * lambda_vec)

    return f, g.reshape(-1, 1), H

# -------------------------------------------------------------
# GO-MTL 
# -------------------------------------------------------------
def GO_MTL(X_train, Y_train, k, Maxsteps, lambda_, mu, O):
    T = len(X_train)
    D = X_train[0].shape[1]

    # Step 1: Train independent models
    W = np.zeros((D, T))

    if O == 'R':
        baseObj = lambda w, X, y: squared_error(w, X, y)
    else:
        baseObj = lambda w, X, y: logistic_loss(w, X, y)


    for t in range(T):
        funObj = lambda w, t=t: baseObj(w, X_train[t], Y_train[t])
        funObjL2 = lambda w, t=t: penalized_l2(w, funObj, lambda_ * np.ones(D))
        W[:, t] = mainfun.minFunc(funObjL2, np.zeros(D))

    # Step 2: SVD initialization
    U, _, _ = svd(W, full_matrices=False)
    L = U[:, :k] 
    S = np.zeros((k, T))

    # Step 3: Alternating optimization
    for step in range(Maxsteps):

        # --- Solve for S (sparse coefficients) ---
        for t in range(T):
            XtL = X_train[t] @ L
            #print("XtL shape:", XtL.shape)

            funObj = lambda s, XtL=XtL, t=t: baseObj(s, XtL, Y_train[t])  # ← fix
            # L1 solver
            S[:, t] = mainfun.L1General2_PSSgb(funObj, S[:, t], mu * np.ones(k))

        # --- Solve for L (latent basis tasks) ---
        if O == 'R':
            A = np.eye(D * k) * lambda_
            B = np.zeros(D * k)

            for t in range(T):
                Xt = X_train[t]
                Nt = Xt.shape[0]
                s = S[:, t]

                A += (1 / Nt) * np.kron(np.outer(s, s), Xt.T @ Xt)
                B += (1 / Nt) * np.kron(s, (Xt.T @ Y_train[t]).flatten())

            L = solve(A, B).reshape(D, k, order='F')

        else:
            # Logistic case — gradient update 
            deltaL = 2 * lambda_ * L

            for t in range(T):
                Xt = X_train[t]
                Nt = Xt.shape[0]
                s = S[:, t]
                z = Xt @ (L @ s)
                pred = 1 / (1 + np.exp(-z))
                deltaL -= (1 / Nt) * Xt.T @ (Y_train[t] - pred)[:, None] @ s[None, :]

            L -= 0.005 * deltaL

    W = L @ S
    return L, S, W
