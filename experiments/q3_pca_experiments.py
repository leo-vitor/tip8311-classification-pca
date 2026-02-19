# experiments/q3_pca_experiments.py

import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import get_data
from src.pca_analysis import PCA
from src.cqg_classifier import CQGClassifier
from src.dmp_classifier import DMPClassifier


def train_test_split(X, y, test_size=0.3, seed=42):
    np.random.seed(seed)
    idx = np.random.permutation(len(X))

    n_test = int(test_size * len(X))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def evaluate_with_pca(X, y, classifier, q, n_runs=100):
    global_scores = []
    class_scores = {c: [] for c in np.unique(y)}

    for i in range(n_runs):
        Xtr, Xte, ytr, yte = train_test_split(X, y, seed=i)

        # PCA ajustado somente no treino
        pca = PCA().fit(Xtr)

        Xtr_p = pca.transform(Xtr, q)
        Xte_p = pca.transform(Xte, q)

        clf = classifier()
        clf.fit(Xtr_p, ytr)

        ypred = clf.predict(Xte_p)

        # acurácia global
        global_scores.append(np.mean(ypred == yte))

        # acurácia por classe
        for c in class_scores:
            idx = (yte == c)
            if np.sum(idx) > 0:
                class_scores[c].append(np.mean(ypred[idx] == yte[idx]))

    results = {
        "global": (np.mean(global_scores), np.std(global_scores))
    }

    for c in class_scores:
        results[f"class_{c}"] = (
            np.mean(class_scores[c]),
            np.std(class_scores[c])
        )

    return results




def run():
    print("\n==============================")
    print(" Q3 - PCA E REDUÇÃO DE DIMENSÃO")
    print("==============================")

    X, y, _ = get_data(num_sensors=24)

    pca = PCA().fit(X)
    cumvar = pca.cumulative_variance()

    q = pca.select_q(0.95)
    print(f"Número de componentes selecionado (q) para 95% VE: {q}")

    plt.figure()
    plt.plot(range(1, len(cumvar) + 1), cumvar)
    plt.xlabel("Número de Componentes (q)")
    plt.ylabel("Variância Explicada Acumulada")
    plt.title("Curva VE(q)")
    plt.grid(True)
    plt.show()

    # função auxiliar corretamente indentada
    def print_results(name, results):
        print(f"\n{name}")
        print(f"Global: {results['global'][0]:.4f} ± {results['global'][1]:.4f}")
        for key in sorted(results.keys()):
            if key != "global":
                print(f"{key}: {results[key][0]:.4f} ± {results[key][1]:.4f}")

    print("\n--- CQG com PCA ---")
    cqg_results = evaluate_with_pca(X, y, CQGClassifier, q)
    print_results("CQG + PCA", cqg_results)

    print("\n--- DMP com PCA ---")
    dmp_results = evaluate_with_pca(X, y, DMPClassifier, q)
    print_results("DMP + PCA", dmp_results)

    return {
        "CQG_PCA": cqg_results,
        "DMP_PCA": dmp_results,
    }




if __name__ == "__main__":
    run()
