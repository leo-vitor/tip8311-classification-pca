# src/cqg_classifier.py

import numpy as np
from numpy.linalg import inv, slogdet
from collections import defaultdict


class CQGClassifier:
    """
    Classificador Quadrático Gaussiano.
    """

    def __init__(self):
        self.classes_ = None
        self.means_ = {}
        self.covariances_ = {}
        self.inv_covariances_ = {}
        self.log_dets_ = {}
        self.priors_ = {}

    def fit(self, X, y):
        """
        Estima parâmetros do modelo.
        """
        self.classes_ = np.unique(y)
        N = X.shape[0]

        for c in self.classes_:
            Xc = X[y == c]

            # Média
            mu = np.mean(Xc, axis=0)

            # Covariância (divisão por N)
            Xc_centered = Xc - mu
            Sigma = (Xc_centered.T @ Xc_centered) / Xc.shape[0]

            # Armazenamento
            self.means_[c] = mu
            self.covariances_[c] = Sigma
            self.inv_covariances_[c] = inv(Sigma)

            sign, logdet = slogdet(Sigma)
            self.log_dets_[c] = logdet

            # Prior
            self.priors_[c] = Xc.shape[0] / N

        return self

    def _discriminant(self, X, c):
        """
        Avalia g_k(x) para uma classe.
        """
        mu = self.means_[c]
        inv_cov = self.inv_covariances_[c]
        log_det = self.log_dets_[c]
        prior = self.priors_[c]

        diff = X - mu
        quad_term = np.sum(diff @ inv_cov * diff, axis=1)

        return (
            -0.5 * quad_term
            -0.5 * log_det
            + np.log(prior)
        )

    def predict(self, X):
        """
        Classifica padrões.
        """
        scores = np.column_stack([
            self._discriminant(X, c) for c in self.classes_
        ])

        return self.classes_[np.argmax(scores, axis=1)]

    def score(self, X, y):
        """
        Retorna acurácia.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
