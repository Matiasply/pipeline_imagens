from typing import Optional, Sequence

import numpy as np


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

    if not face_landmarks:
        return None

    h, w, _ = frame.shape
    coords = []
    for lm in face_landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        coords.append((x, y))

    valid = [coords[i] for i in lip_indices if 0 <= i < len(coords)]
    if not valid:
        return None

    xs = [p[0] for p in valid]
    ys = [p[1] for p in valid]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    pad = int(0.4 * max(xmax - xmin, ymax - ymin))
    xmin_p = max(xmin - pad, 0)
    ymin_p = max(ymin - pad, 0)
    xmax_p = min(xmax + pad, w - 1)
    ymax_p = min(ymax + pad, h - 1)

    if xmin_p >= xmax_p or ymin_p >= ymax_p:
        return None

    return frame[ymin_p:ymax_p, xmin_p:xmax_p]
