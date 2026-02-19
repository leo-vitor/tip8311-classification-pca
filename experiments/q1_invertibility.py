# experiments/q1_invertibility.py

from src.data_loader import get_data
from src.covariance_analysis import analyze_invertibility, covariance_by_class

def run():
    main()
def print_results(title, results):
    print("=" * 60)
    print(title)
    print("=" * 60)

    print(f"Posto da matriz      : {results['rank']}")
    print(f"rcond                : {results['rcond']:.3e}")
    print(f"Inversível?           : {results['invertible']}")
    print(f"Bem condicionada?     : {results['well_conditioned']}")
    print()


def main():
    # -------------------------
    # 1. Carregamento dos dados
    # -------------------------
    X, y, _ = get_data(num_sensors=24)

    # -------------------------
    # 2. Covariância global
    # -------------------------
    global_results = analyze_invertibility(X)
    print_results("Matriz de Covariância Global", global_results)

    # -------------------------
    # 3. Covariância por classe
    # -------------------------
    class_results = covariance_by_class(X, y)

    for cls, results in class_results.items():
        print_results(f"Classe: {cls}", results)


if __name__ == "__main__":
    run()
