# main.py

from experiments import q1_invertibility
from experiments import q2_classification
from experiments import q3_pca_experiments


def main():
    print("\n==============================")
    print(" TRABALHO COMPUTACIONAL 3")
    print(" Reconhecimento de Padrões")
    print("==============================\n")

    # -------------------------
    # Q1 - Invertibilidade
    # -------------------------
    print(">>> Executando Q1: Análise de Invertibilidade\n")
    q1_invertibility.run()

    # -------------------------
    # Q2 - Classificação
    # -------------------------
    print(">>> Executando Q2: Classificação CQG e DMP\n")
    q2_classification.run()

    # -------------------------
    # Q3 - PCA
    # -------------------------
    print(">>> Executando Q3: Análise com PCA\n")
    q3_pca_experiments.run()


if __name__ == "__main__":
    main()
