from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPO = PROJECT_ROOT / "third_party" / "LipNet-PyTorch"

LETTERS = [" ", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]


def _ensure_official_repo_on_path() -> Path:
    if not OFFICIAL_REPO.exists():
        raise FileNotFoundError(
            "O repositório oficial LipNet não foi encontrado em "
            f"{OFFICIAL_REPO}. Clone-o em third_party/LipNet-PyTorch."
        )
    if str(OFFICIAL_REPO) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_REPO))
    return OFFICIAL_REPO


def _resolve_weights_path(weights_path: Optional[Path | str] = None) -> Optional[Path]:
    repo = _ensure_official_repo_on_path()
    if weights_path is not None:
        candidate = Path(weights_path)
        if candidate.exists():
            return candidate
    candidates = [
        repo / "pretrain" / "LipNet_unseen_loss_0.44562849402427673_wer_0.1332580699113564_cer_0.06796452465503355.pt",
        repo / "pretrain" / "LipNet_overlap_loss_0.07664558291435242_wer_0.04644484056248762_cer_0.019676921477851092.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_lipnet_model(weights_path: Optional[Path | str] = None, device: str = "cpu") -> nn.Module:
    _ensure_official_repo_on_path()
    from model import LipNet

    model = LipNet()
    model.to(device)
    model.eval()

    resolved_path = _resolve_weights_path(weights_path)
    if resolved_path is not None:
        try:
            checkpoint = torch.load(resolved_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(resolved_path, map_location=device)

        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("state_dict", checkpoint)
            if isinstance(state_dict, dict):
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict, strict=False)

    return model


def _read_video_frames(video_path: Path | str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {video_path}")

    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def video_to_lipnet_tensor(video_path: Path | str, target_size: tuple[int, int] = (128, 64)) -> torch.Tensor:
    frames = _read_video_frames(video_path)
    if not frames:
        raise ValueError(f"O vídeo está vazio: {video_path}")

    processed = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, target_size, interpolation=cv2.INTER_LANCZOS4)
        processed.append(resized.astype(np.float32) / 255.0)

    video = np.stack(processed, axis=0)  # T, H, W, 3
    video = np.transpose(video, (0, 3, 1, 2))  # T, 3, H, W
    video = video.transpose(1, 0, 2, 3)  # 3, T, H, W
    video = np.expand_dims(video, axis=0)  # 1, 3, T, H, W
    return torch.from_numpy(video)


def _decode_lipnet_logits(logits: torch.Tensor) -> str:
    logits = logits.detach().cpu()
    predicted = logits.argmax(dim=-1)
    decoded: list[str] = []
    previous = -1
    for token in predicted.tolist():
        if token >= 1 and token != previous:
            decoded.append(LETTERS[token - 1])
        previous = token
    return "".join(decoded).strip()


def predict_text_from_video(
    video_path: Path | str,
    model: Optional[nn.Module] = None,
    device: str = "cpu",
    weights_path: Optional[Path | str] = None,
) -> dict:
    tensor = video_to_lipnet_tensor(video_path)
    tensor = tensor.to(device)

    if model is None:
        model = load_lipnet_model(weights_path=weights_path, device=device)
    model.eval()

    with torch.no_grad():
        logits = model(tensor)
        decoded = _decode_lipnet_logits(logits[0])

    return {
        "predicted_text": decoded,
        "shape": list(tensor.shape),
        "logits_shape": list(logits.shape),
    }
