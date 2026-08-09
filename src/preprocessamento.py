import cv2
import numpy as np


def filtro_gaussiano(imagem: np.ndarray, ksize: tuple = (5, 5)) -> np.ndarray:
    """
    Aplica o filtro Gaussiano para reduzir ruídos.
    """
    return cv2.GaussianBlur(imagem, ksize, 0)


def filtro_media(imagem: np.ndarray, ksize: tuple = (3, 3)) -> np.ndarray:
    """
    Aplica o filtro da média para suavizar a imagem.
    """
    return cv2.blur(imagem, ksize)


def aplicar_clahe(imagem: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Corrige iluminação usando CLAHE.
    Para imagens coloridas, converte para YCrCb e aplica CLAHE apenas no canal Y.
    """
    if imagem is None:
        return imagem

    if len(imagem.shape) == 2 or imagem.shape[2] == 1:
        # imagem em escala de cinza
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(imagem)

    # imagem colorida: converte para YCrCb e equaliza canal Y
    ycrcb = cv2.cvtColor(imagem, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    y_eq = clahe.apply(y)
    ycrcb_eq = cv2.merge((y_eq, cr, cb))
    return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)


def reduzir_ruido(imagem: np.ndarray, h: int = 10, hColor: int = 10, templateWindowSize: int = 7, searchWindowSize: int = 21) -> np.ndarray:
    """
    Reduz ruído usando Non-local Means denoising.
    Detecta automaticamente se a imagem é colorida.
    """
    if imagem is None:
        return imagem

    if len(imagem.shape) == 2 or (len(imagem.shape) == 3 and imagem.shape[2] == 1):
        # grayscale
        return cv2.fastNlMeansDenoising(imagem, None, h, templateWindowSize, searchWindowSize)

    # color
    return cv2.fastNlMeansDenoisingColored(imagem, None, h, hColor, templateWindowSize, searchWindowSize)


def normalizar_imagem(imagem: np.ndarray, to_float: bool = False) -> np.ndarray:
    """
    Normaliza a imagem.
    - Se to_float for True, retorna float32 com valores em [0, 1].
    - Caso contrário, retorna uint8 com faixa [0, 255].
    """
    if imagem is None:
        return imagem

    if to_float:
        img = imagem.astype(np.float32) / 255.0
        return img

    # garantir uint8
    if imagem.dtype == np.uint8:
        return imagem
    img = np.clip(imagem, 0.0, 1.0)
    img = (img * 255.0).astype(np.uint8)
    return img


def preprocessar(imagem: np.ndarray,
                 aplicar_illuminacao: bool = True,
                 reduzir_ruido_flag: bool = True,
                 aplicar_suavizacao: bool = True,
                 normalize_to_float: bool = False) -> np.ndarray:
    """
    Executa etapas de pré-processamento em ordem pensada para extração de lábios.

    Passos recomendados (padrão):
    1. Correção de iluminação (CLAHE)
    2. Redução de ruído (Non-local Means)
    3. Suavização (Gaussiano então Média) - opcional
    4. Normalização de intensidade (opcional: retorna float32 em [0,1])

    Parâmetros:
    - aplicar_illuminacao: aplica CLAHE para melhorar contraste local
    - reduzir_ruido_flag: aplica denoising (rápido e eficaz)
    - aplicar_suavizacao: aplica gaussian + média para remoção de artefatos residuais
    - normalize_to_float: se True retorna float32 em [0,1], caso contrário uint8
    """
    if imagem is None:
        return imagem

    img = imagem.copy()

    # Correção de iluminação
    if aplicar_illuminacao:
        img = aplicar_clahe(img)

    # Redução de ruído
    if reduzir_ruido_flag:
        img = reduzir_ruido(img)

    # Suavização adicional
    if aplicar_suavizacao:
        img = filtro_gaussiano(img)
        img = filtro_media(img)

    # Se imagem for colorida e normalize_to_float for False, manter uint8
    img = normalizar_imagem(img, to_float=normalize_to_float)

    return img
