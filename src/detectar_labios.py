from typing import Optional, Sequence

import cv2
import numpy as np


LIPS_OUTPUT_SIZE = (96, 96)
LIPS_SMOOTHING_ALPHA = 0.8
_previous_lips_box = None


def crop_lips(frame: np.ndarray, face_landmarks: Sequence) -> Optional[np.ndarray]:
    """Recorta a região dos lábios a partir dos landmarks da face.

    Args:
        frame: imagem RGB como numpy array.
        face_landmarks: sequência de objetos com atributos .x e .y normalizados (0..1).

    Retorna:
        Recorte RGB como numpy array ou None se não for possível.
    """
    # índices do contorno externo dos lábios no Face Mesh (MediaPipe)
    lip_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

    # Usa variável global para manter o estado entre frames
    global _previous_lips_box

    if not face_landmarks:
        _previous_lips_box = None
        return None

    # Calcula coordenadas dos landmarks em pixels
    h, w, _ = frame.shape
    coords = []
    for lm in face_landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        coords.append((x, y))

    valid = [coords[i] for i in lip_indices if 0 <= i < len(coords)]
    if not valid:
        _previous_lips_box = None
        return None

    # Calcula a bounding box dos lábios com padding
    xs = [p[0] for p in valid]
    ys = [p[1] for p in valid]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    pad = int(0.4 * max(xmax - xmin, ymax - ymin))
    xmin_p = max(xmin - pad, 0)
    ymin_p = max(ymin - pad, 0)
    xmax_p = min(xmax + pad, w - 1)
    ymax_p = min(ymax + pad, h - 1)

    # Verifica se a bounding box é válida
    if xmin_p >= xmax_p or ymin_p >= ymax_p:
        _previous_lips_box = None
        return None

    # Suaviza a bounding box usando a média ponderada com a última posição conhecida
    current_box = np.array([xmin_p, ymin_p, xmax_p, ymax_p], dtype=np.float32)
    if _previous_lips_box is None:
        # Se não houver posição anterior, usa a posição atual
        smoothed_box = current_box
    else:
        # Calcula a média ponderada entre a posição atual e a última posição conhecida
        previous_box = np.array(_previous_lips_box, dtype=np.float32)
        smoothed_box = (
            LIPS_SMOOTHING_ALPHA * previous_box
            + (1.0 - LIPS_SMOOTHING_ALPHA) * current_box
        )

    xmin_p, ymin_p, xmax_p, ymax_p = [int(round(value)) for value in smoothed_box]
    xmin_p = max(min(xmin_p, w - 1), 0)
    ymin_p = max(min(ymin_p, h - 1), 0)
    xmax_p = max(min(xmax_p, w - 1), 0)
    ymax_p = max(min(ymax_p, h - 1), 0)

    # Verifica se a bounding box suavizada é válida
    if xmin_p >= xmax_p or ymin_p >= ymax_p:
        _previous_lips_box = None
        return None

    # Atualiza a variável global com a bounding box suavizada
    _previous_lips_box = (xmin_p, ymin_p, xmax_p, ymax_p)

    # Recorta a região dos lábios e redimensiona para o tamanho desejado
    lips_crop = frame[ymin_p:ymax_p, xmin_p:xmax_p]
    crop_h, crop_w = lips_crop.shape[:2]
    if crop_h == 0 or crop_w == 0:
        _previous_lips_box = None
        return None

    # Redimensiona o recorte para o tamanho desejado mantendo a proporção
    target_w, target_h = LIPS_OUTPUT_SIZE
    scale = min(target_w / crop_w, target_h / crop_h)
    resized_w = max(1, int(round(crop_w * scale)))
    resized_h = max(1, int(round(crop_h * scale)))
    resized_crop = cv2.resize(lips_crop, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    # Cria uma imagem preta do tamanho desejado e centraliza o recorte redimensionado
    canvas = np.zeros((target_h, target_w, frame.shape[2]), dtype=frame.dtype)
    y_offset = (target_h - resized_h) // 2
    x_offset = (target_w - resized_w) // 2
    canvas[y_offset:y_offset + resized_h, x_offset:x_offset + resized_w] = resized_crop
    return canvas
