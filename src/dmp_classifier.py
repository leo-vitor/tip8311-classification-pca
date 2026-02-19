# src/dmp_classifier.py

import numpy as np


class DMPClassifier:
    """
    Classificador de Distância Mínima à Média.
    """

    def __init__(self):
        self.classes_ = None
        self.means_ = {}

    def fit(self, X, y):
        """
        Estima as médias de cada classe.
        """
        self.classes_ = np.unique(y)

        for c in self.classes_:
            self.means_[c] = np.mean(X[y == c], axis=0)

        return self

    def predict(self, X):
        """
        Classifica padrões pela menor distância euclidiana à média.
        """
        distances = []

        for c in self.classes_:
            diff = X - self.means_[c]
            dist = np.sum(diff ** 2, axis=1)
            distances.append(dist)

        distances = np.column_stack(distances)

        return self.classes_[np.argmin(distances, axis=1)]

    def score(self, X, y):
        """
        Retorna acurácia.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
