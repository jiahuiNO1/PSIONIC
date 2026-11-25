import numpy as np
from scipy.linalg import cholesky

def isLegal(v):
    v = np.asarray(v, dtype=complex)
    return not (np.any(np.isnan(v)) or np.any(np.isinf(v)) or np.any(np.imag(v) != 0))


def polyinterp(points, doPlot=0, xminBound=None, xmaxBound=None):
    points = np.array(points, dtype=float)
    nPoints = points.shape[0]
    
    # Count known f and g values to determine polynomial order
    order = int(np.sum(~np.isnan(points[:, 1:3]))) - 1
    
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0])
    if xminBound is None:
        xminBound = xmin
    if xmaxBound is None:
        xmaxBound = xmax

    # Fast path: cubic interpolation of 2 points with f and g known 
    if nPoints == 2 and order == 3 and doPlot == 0:
        minPos = int(np.argmin(points[:, 0]))
        notMinPos = 1 - minPos
        d1 = (points[minPos, 2] + points[notMinPos, 2]
              - 3 * (points[minPos, 1] - points[notMinPos, 1])
              / (points[minPos, 0] - points[notMinPos, 0]))
        d2 = np.sqrt(d1**2 - points[minPos, 2] * points[notMinPos, 2])
        if np.isreal(d2):
            t = (points[notMinPos, 0]
                 - (points[notMinPos, 0] - points[minPos, 0])
                 * ((points[notMinPos, 2] + d2 - d1)
                    / (points[notMinPos, 2] - points[minPos, 2] + 2 * d2)))
            minPos = float(np.real(min(max(t, xminBound), xmaxBound)))
        else:
            minPos = (xmaxBound + xminBound) / 2
        return minPos

    # Fallback: bisection (should never be reached under Newton defaults)
    return float((xmaxBound + xminBound) / 2)



def ArmijoBacktrack(funObj, x, t, d, f, fr, g, gtd,
                    c1, LS_interp, LS_multi, progTol, debug, doPlot, saveHessianComp):

    f_new, g_new = funObj(x + t * d)[:2]
    g_new = g_new.flatten()
    funEvals = 1
    f_prev = None
    t_prev = None
    g_prev = None

    while f_new > fr + c1 * t * gtd or not isLegal(f_new):
        temp = t

        if LS_interp == 0 or not isLegal(f_new):
            # Fixed backtracking: halve step
            t = 0.5 * t

        elif LS_interp == 1 or not isLegal(g_new):
            # Quadratic/cubic interpolation using only function values
            if funEvals < 2 or LS_multi == 0 or f_prev is None or not isLegal(f_prev):
                t = polyinterp(
                    np.array([[0, f, gtd],
                              [temp, f_new, np.nan * 1j]]),
                    doPlot, 0, temp
                )
            else:
                t = polyinterp(
                    np.array([[0, f, gtd],
                              [temp, f_new, np.nan * 1j],
                              [t_prev, f_prev, np.nan * 1j]]),
                    doPlot, 0, temp
                )

        else:
            # Cubic/quintic interpolation using function values and gradients
            if funEvals < 2 or LS_multi == 0 or f_prev is None or not isLegal(f_prev):
                t = polyinterp(
                    np.array([[0, f, gtd],
                              [temp, f_new, g_new @ d]]),
                    doPlot, 0, temp
                )
            elif g_prev is None or not isLegal(g_prev):
                t = polyinterp(
                    np.array([[0, f, gtd],
                              [temp, f_new, g_new @ d],
                              [t_prev, f_prev, np.nan * 1j]]),
                    doPlot, 0, temp
                )
            else:
                t = polyinterp(
                    np.array([[0, f, gtd],
                              [temp, f_new, g_new @ d],
                              [t_prev, f_prev, g_prev @ d]]),
                    doPlot, 0, temp
                )

        # Clamp step size change
        if t < temp * 1e-3:
            t = temp * 1e-3
        elif t > temp * 0.6:
            t = temp * 0.6

        # Store previous point if doing multi-point interpolation
        if LS_multi:
            f_prev = f_new
            t_prev = temp
            if LS_interp == 2:
                g_prev = g_new.copy()

        # Evaluate new point (skip H during line search if saveHessianComp=1)
        f_new, g_new = funObj(x + t * d)[:2]
        g_new = g_new.flatten()
        funEvals += 1

        if np.max(np.abs(t * d)) <= progTol:
            t = 0
            f_new = f
            g_new = g.flatten()
            break

    # Compute H at final point if requested
    if saveHessianComp and funEvals > 1:
        f_new, g_new, H = funObj(x + t * d)
        g_new = g_new.flatten()
        funEvals += 1
    else:
        H = None

    x_new = x + t * d
    return t, x_new, f_new, g_new, funEvals, H


def funObj_fg(funObj, x):
    """Helper: call funObj but only use f and g, discard H."""
    result = funObj(x)
    return result[0], result[1]


def WolfeLineSearch(funObj, x, t, d, f, g, gtd, c1, c2, LS_interp, LS_multi,
                    maxLS, progTol, debug, doPlot, saveHessianComp):

    f_new, g_new = funObj_fg(funObj, x + t * d)
    funEvals = 1
    g_new = g_new.flatten()
    gtd_new = g_new @ d

    LSiter = 0
    t_prev = 0
    f_prev = f
    g_prev = g.flatten()
    gtd_prev = gtd
    done = False

    while LSiter < maxLS:

        if not isLegal(f_new) or not isLegal(g_new):
            t = (t + t_prev) / 2
            t, x_new, f_new, g_new, armijoFunEvals, _ = ArmijoBacktrack(
                funObj, x, t, d, f, f, g, gtd,
                c1, LS_interp, LS_multi, progTol, debug, doPlot, saveHessianComp
            )
            funEvals += armijoFunEvals
            return t, f_new, g_new.flatten(), funEvals, None  # H handled below

        if f_new > f + c1 * t * gtd or (LSiter > 1 and f_new >= f_prev):
            bracket = [t_prev, t]
            bracketFval = [f_prev, f_new]
            bracketGval = [g_prev.copy(), g_new.copy()]
            break
        elif abs(gtd_new) <= -c2 * gtd:
            bracket = [t]
            bracketFval = [f_new]
            bracketGval = [g_new.copy()]
            done = True
            break
        elif gtd_new >= 0:
            bracket = [t_prev, t]
            bracketFval = [f_prev, f_new]
            bracketGval = [g_prev.copy(), g_new.copy()]
            break

        temp = t_prev
        t_prev = t
        minStep = t + 0.01 * (t - temp)
        maxStep = t * 10

        if LS_interp <= 1:
            t = maxStep
        elif LS_interp == 2:
            t = polyinterp(
                np.array([[temp, f_prev, gtd_prev],
                          [t,    f_new,  gtd_new]]),
                doPlot, minStep, maxStep
            )

        f_prev = f_new
        g_prev = g_new.copy()
        gtd_prev = gtd_new

        f_new, g_new = funObj_fg(funObj, x + t * d)
        g_new = g_new.flatten()
        funEvals += 1
        gtd_new = g_new @ d
        LSiter += 1

    if LSiter == maxLS:
        bracket = [0, t]
        bracketFval = [f, f_new]
        bracketGval = [g.flatten().copy(), g_new.copy()]

    # Zoom phase
    insufProgress = 0

    while not done and LSiter < maxLS:

        f_LO = min(bracketFval)
        LOpos = bracketFval.index(f_LO)
        HIpos = 1 - LOpos

        if LS_interp <= 1 or not isLegal(bracketFval) or not isLegal(np.array(bracketGval)):
            t = float(np.mean(bracket))
        elif LS_interp == 2:
            t = polyinterp(
                np.array([[bracket[0], bracketFval[0], bracketGval[0] @ d],
                          [bracket[1], bracketFval[1], bracketGval[1] @ d]]),
                doPlot
            )

        bmax = max(bracket)
        bmin = min(bracket)
        brange = bmax - bmin
        if brange > 0 and min(bmax - t, t - bmin) / brange < 0.1:
            if insufProgress or t >= bmax or t <= bmin:
                if abs(t - bmax) < abs(t - bmin):
                    t = bmax - 0.1 * brange
                else:
                    t = bmin + 0.1 * brange
                insufProgress = 0
            else:
                insufProgress = 1
        else:
            insufProgress = 0

        f_new, g_new = funObj_fg(funObj, x + t * d)
        g_new = g_new.flatten()
        funEvals += 1
        gtd_new = g_new @ d
        LSiter += 1

        if f_new > f + c1 * t * gtd or f_new >= f_LO:
            bracket[HIpos] = t
            bracketFval[HIpos] = f_new
            bracketGval[HIpos] = g_new.copy()
        else:
            if abs(gtd_new) <= -c2 * gtd:
                done = True
            elif gtd_new * (bracket[HIpos] - bracket[LOpos]) >= 0:
                bracket[HIpos] = bracket[LOpos]
                bracketFval[HIpos] = bracketFval[LOpos]
                bracketGval[HIpos] = bracketGval[LOpos].copy()
            bracket[LOpos] = t
            bracketFval[LOpos] = f_new
            bracketGval[LOpos] = g_new.copy()

        if not done and abs((bracket[0] - bracket[1]) * gtd_new) < progTol:
            break

    f_LO = min(bracketFval)
    LOpos = bracketFval.index(f_LO)
    t = bracket[LOpos]
    f_new = bracketFval[LOpos]
    g_new = bracketGval[LOpos].copy()

    # Matches MATLAB: if nargout==5 && funEvals > 1 && saveHessianComp
    if saveHessianComp and funEvals > 1:
        f_new, g_new, H = funObj(x + t * d)
        g_new = g_new.flatten()
        funEvals += 1
    else:
        H = None

    return t, f_new, g_new, funEvals, H


def minFunc(funObj, x0):
    maxIter = 500
    maxFunEvals = 1000
    optTol = 1e-5
    progTol = 1e-9
    c1 = 1e-4
    c2 = 0.9
    LS_interp = 2
    LS_multi = 0
    LS_saveHessianComp = 1
    maxLS = 25
    debug = 0
    doPlot = 0

    x = x0.flatten()
    p = len(x)

    f, g, H = funObj(x)
    g = g.flatten()
    funEvals = 1

    if np.max(np.abs(g)) <= optTol:
        return x

    for i in range(1, maxIter + 1):

        try:
            R = cholesky(H, lower=False)
            d = -np.linalg.solve(R, np.linalg.solve(R.T, g))
        except np.linalg.LinAlgError:
            eigvals = np.linalg.eigvalsh(H)
            H = H + np.eye(p) * max(0, 1e-12 - np.min(eigvals))
            d = -np.linalg.solve(H, g)

        gtd = g @ d
        if gtd > -progTol:
            break

        t = 1.0
        f_old = f

        t, f, g, LSfunEvals, H = WolfeLineSearch(
            funObj, x, t, d, f, g, gtd,
            c1, c2, LS_interp, LS_multi,
            maxLS, progTol, debug, doPlot, LS_saveHessianComp
        )
        funEvals += LSfunEvals
        x = x + t * d       # update x before convergence checks
        g = g.flatten()

        if np.max(np.abs(g)) <= optTol:
            break
        if np.max(np.abs(t * d)) <= progTol:
            break
        if abs(f - f_old) < progTol:
            break
        if funEvals >= maxFunEvals:
            break

    return x




#############
def pseudoGrad(funObj, w, lambda_val):
    result = funObj(w)
    f, g = result[0], result[1]   # ← safely take first two
    f = float(f) + float(np.sum(lambda_val * np.abs(w)))
    g = g.reshape(-1, 1)
    lambda_val = lambda_val.reshape(-1, 1)
    pGrad = np.zeros_like(g)
    pGrad[g < -lambda_val] = g[g < -lambda_val] + lambda_val[g < -lambda_val]
    pGrad[g > lambda_val] = g[g > lambda_val] - lambda_val[g > lambda_val]
    nonZero = np.logical_or(w.ravel() != 0, lambda_val.ravel() == 0)
    pGrad[nonZero] = g[nonZero] + lambda_val[nonZero] * np.sign(w.ravel()[nonZero]).reshape(-1, 1)
    return f, pGrad


def orthantProject(w, xi):
    w = w.ravel().copy()
    xi = xi.ravel()
    w[np.sign(w) != xi] = 0
    return w


def lbfgs(g, s, y, Hdiag):
    if s.ndim == 1:
        s = s.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    p, k = s.shape
    ro = np.array([1.0 / (y[:, i] @ s[:, i]) for i in range(k)])
    q = np.zeros((p, k + 1))
    r = np.zeros((p, k + 1))
    al = np.zeros(k)
    be = np.zeros(k)
    q[:, k] = g.ravel()
    for i in range(k - 1, -1, -1):
        al[i] = ro[i] * (s[:, i] @ q[:, i + 1])
        q[:, i] = q[:, i + 1] - al[i] * y[:, i]
    r[:, 0] = Hdiag * q[:, 0]
    for i in range(k):
        be[i] = ro[i] * (y[:, i] @ r[:, i])
        r[:, i + 1] = r[:, i] + s[:, i] * (al[i] - be[i])
    return r[:, k]

    

def L1General2_PSSgb(funObj, w, lambda_val, options=None):
    if options is None:
        options = {}
    
    w = w.ravel().copy()    # ← ensure 1D from the start

    lambda_val = lambda_val.reshape(-1, 1)
    verbose = options.get('verbose', 0)
    optTol = options.get('optTol', 1e-5)
    progTol = options.get('progTol', 1e-9)
    maxIter = options.get('maxIter', 500)
    suffDec = options.get('suffDec', 1e-4)
    corrections = options.get('corrections', 100)
    Dtype = options.get('Dtype', 1)
    quadraticInit = options.get('quadraticInit', 0)

    p = len(w)

    if verbose:
        print('%6s %6s %12s %12s %12s %6s' % ('Iter', 'fEvals', 'stepLen', 'fVal', 'optCond', 'nnz'))

    # Evaluate Initial Point
    f, g = pseudoGrad(funObj, w, lambda_val)
    g = g.ravel()
    g_old = g
    funEvals = 1

    # Check optimality
    optCond = np.max(np.abs(g))
    if optCond < optTol:
        if verbose:
            print('First-order optimality satisfied at initial point')
        return w

    # Main Loop
    Y = np.zeros((p, 0))
    S = np.zeros((p, 0))
    sigma = 1
    t = min(1, 1 / np.sum(np.abs(g)))
    w_old = w.copy()

    for i in range(maxIter):
        # Compute active set
        W = np.logical_or(lambda_val.ravel() == 0, w.ravel() != 0)
        A = np.logical_not(W)

        # Compute direction
        d = np.zeros(p)

        if i == 0:
            d = -g.ravel()
        else:
            y = g - g_old
            s = w.reshape(-1, 1) - w_old.reshape(-1, 1)

            correctionsStored = Y.shape[1]
            if correctionsStored < corrections:
                Y = np.column_stack((Y, y))
                S = np.column_stack((S, s))
            else:
                Y = np.c_[Y[:, 1:corrections], y]
                S = np.c_[S[:, 1:corrections], s]

            ys = float((y.T @ s).item())
            if ys > 1e-10:
                sigma = ys / float((y.T @ y).item())

            W_flat = W.ravel()
            curvSat = np.sum(Y[W_flat, :] * S[W_flat, :], axis=0) > 1e-10

            if curvSat.any():
                d[W_flat] = lbfgs(-g.ravel()[W_flat],
                                   S[W_flat, :][:, curvSat],
                                   Y[W_flat, :][:, curvSat],
                                   sigma)
            else:
                continue

            
            if Dtype == 0:
                D = min(1, 1 / np.sum(np.abs(g)))
            else:
                D = sigma
            d[A.ravel()] = -D * g.ravel()[A.ravel()]

            t = 1

        f_old = f
        g_old = g
        w_old = w.copy()

        # Compute desired orthant
        xi = np.sign(w)
        xi[w == 0] = np.sign(-g[w == 0])

        # Compute directional derivative and check progress
        gtd = float(g.ravel() @ d.ravel())
        if gtd > -progTol:
            if verbose:
                print('Directional derivative below progTol')
            break

        # quadraticInit
        if quadraticInit:
            if i > 1:
                t = min(1, 2 * (f - f_prev) / gtd)
            f_prev = f

        # Compute projected point
        w_candidate = w + t * d.ravel()  # Keep as 1D
        w_new = orthantProject(w_candidate, xi)
        f_new, g_new = pseudoGrad(funObj, w_new, lambda_val)
        g_new = g_new.ravel()
        funEvals += 1

        # Line search along projection arc
        while f_new > f + suffDec * float(g.ravel() @ (w_new.ravel() - w.ravel())) or not isLegal(f_new):
            t_old = t
            if verbose:
                print('Backtracking...')
            if not isLegal(f_new):
                if verbose:
                    print('Halving Step Size')
                t = 0.5 * t
            else:
                t = polyinterp(np.array([[0, f, gtd],
                                         [t, f_new, float(g_new.ravel() @ d.ravel())]]))

            # Adjust if interpolated value near boundary
            if t < t_old * 1e-3:
                if verbose == 3:
                    print('Interpolated value too small, Adjusting')
                t = t_old * 1e-3
            elif t > t_old * 0.6:
                if verbose == 3:
                    print('Interpolated value too large, Adjusting')
                t = t_old * 0.6

            # Check whether step has become too small
            if np.max(np.abs(t * d)) < progTol:
                if verbose:
                    print('Step too small in line search')
                t = 0
                w_new = w.copy()
                f_new = f
                g_new = g
                break

            # Compute projected point
            w_candidate = w + t * d.ravel() 
            w_new = orthantProject(w_candidate, xi)
            f_new, g_new = pseudoGrad(funObj, w_new, lambda_val)
            g_new = g_new.ravel()
            funEvals += 1

        # Take step
        w = w_new.copy()
        f = f_new
        g = g_new

        # Output Log
        if verbose:
            print('%6d %6d %8.5e %8.5e %8.5e %6d' % (
                i, funEvals, t, f, np.max(np.abs(g)), np.count_nonzero(w)))

        # Check Optimality
        if np.max(np.abs(g)) < optTol:
            if verbose:
                print('First-order optimality below optTol')
            break

        # Check for lack of progress
        if np.max(np.abs(t * d)) < progTol or np.abs(f - f_old) < progTol:
            if verbose:
                print('Progress in parameters or objective below progTol')
            break

        # Check for iteration limit
        if funEvals >= maxIter:
            if verbose:
                print('Function evaluations reached maxIter')
            break

    return w.ravel()

