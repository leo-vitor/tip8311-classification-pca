# src/pca_analysis.py

import numpy as np


class PCA:
    """
    Implementação manual de PCA.
    """

    def __init__(self):
        self.mean_ = None
        self.components_ = None
        self.eigenvalues_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        """
        Ajusta PCA aos dados.
        """
        # Centralização
        self.mean_ = np.mean(X, axis=0)
        Xc = X - self.mean_

        # Covariância
        N = X.shape[0]
        C = (Xc.T @ Xc) / N

        # Autodecomposição
        eigvals, eigvecs = np.linalg.eigh(C)

        # Ordenar em ordem decrescente
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        self.eigenvalues_ = eigvals
        self.components_ = eigvecs

        self.explained_variance_ratio_ = eigvals / np.sum(eigvals)

        return self

    def transform(self, X, q):
        """
        Projeta dados nas q primeiras componentes.
        """
        Xc = X - self.mean_
        return Xc @ self.components_[:, :q]

    def cumulative_variance(self):
        return np.cumsum(self.explained_variance_ratio_)

    def select_q(self, threshold=0.95):
        """
        Retorna menor q tal que VE acumulada >= threshold.
        """
        cumvar = self.cumulative_variance()
        return np.searchsorted(cumvar, threshold) + 1
