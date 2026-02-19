# data_loader.py
import pandas as pd
import numpy as np

def get_data(num_sensors=24, encode_labels=True):
    """
    Faz o download e prepara o conjunto de dados
    Wall-Following Robot Navigation (UCI).

    Parâmetros
    ----------
    num_sensors : int
        Número de sensores (4 ou 24).
    encode_labels : bool
        Se True, converte os rótulos para inteiros.

    Retorno
    -------
    X : ndarray (N x p)
        Matriz de atributos.
    y : ndarray (N,)
        Vetor de rótulos de classe.
    labels_map : dict ou None
        Mapeamento rótulo -> inteiro (se encode_labels=True).
    """

    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00194/"
    filename = f"sensor_readings_{num_sensors}.data"
    url = base_url + filename

    try:
        df = pd.read_csv(url, header=None)
    except Exception as e:
        raise RuntimeError(f"Erro ao baixar os dados: {e}")

    # Atributos (sensores)
    X = df.iloc[:, :-1].to_numpy(dtype=float)

    # Rótulos
    y_raw = df.iloc[:, -1].to_numpy()

    if encode_labels:
        classes = np.unique(y_raw)
        labels_map = {c: i for i, c in enumerate(classes)}
        y = np.array([labels_map[label] for label in y_raw])
        return X, y, labels_map

    return X, y_raw, None
