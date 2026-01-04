import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray):
    """Среднеквадратичная ошибка"""
    return np.mean((y_true - y_pred) ** 2)

def mae(y_true: np.ndarray, y_pred: np.ndarray):
    """Средняя абсолютная ошибка"""
    return np.mean(np.abs(y_true - y_pred))