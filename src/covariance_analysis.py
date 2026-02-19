import numpy as np
from numpy.linalg import matrix_rank, cond


def covariance_matrix(X):
    X_centered = X - np.mean(X, axis=0)
    N = X.shape[0]
    return (X_centered.T @ X_centered) / N


def rcond(C):
    return 1.0 / cond(C)


def analyze_invertibility(X, tol=1e-12):
    C = covariance_matrix(X)
    rank = matrix_rank(C, tol=tol)
    rc = rcond(C)

    return {
        "covariance": C,
        "rank": rank,
        "rcond": rc,
        "invertible": rank == C.shape[0],
        "well_conditioned": rc > tol
    }


def covariance_by_class(X, y):
    results = {}
    classes = np.unique(y)

    for c in classes:
        Xc = X[y == c]
        results[c] = analyze_invertibility(Xc)

    return results
