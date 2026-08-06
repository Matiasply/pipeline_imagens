import cv2
import numpy as np


def filtro_gaussiano(imagem: np.ndarray) -> np.ndarray:
    """
    Aplica o filtro Gaussiano para reduzir ruídos.
    """
    return cv2.GaussianBlur(imagem, (5, 5), 0)


def filtro_media(imagem: np.ndarray) -> np.ndarray:
    """
    Aplica o filtro da média para suavizar a imagem.
    """
    return cv2.blur(imagem, (3, 3))


def equalizar_histograma(imagem: np.ndarray) -> np.ndarray:
    """
    Aplica a equalização do histograma.
    A imagem deve estar em escala de cinza.
    """
    return cv2.equalizeHist(imagem)


def preprocessar(imagem: np.ndarray) -> np.ndarray:
    """
    Executa todas as etapas de pré-processamento.

    Ordem:
    1. Filtro Gaussiano
    2. Filtro da Média
    3. Equalização do Histograma
    """

    imagem = filtro_gaussiano(imagem)
    imagem = filtro_media(imagem)
    imagem = equalizar_histograma(imagem)

    return imagem