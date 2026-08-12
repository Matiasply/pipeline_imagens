import json
import os
from typing import List, Tuple

import cv2
import numpy as np


def _check_dependencies():
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        from skimage.metrics import structural_similarity as ssim  # noqa: F401
    except Exception as e:
        raise ImportError(
            "Missing dependencies for evaluation. Install with: \n"
            "pip install torch torchvision scikit-image"
        ) from e


def load_feature_extractor(device: str = "cpu"):
    """
    Carrega ResNet50 pré-treinado e retorna uma feature extractor (model, device).
    Remove a última camada fully-connected e produz embeddings L2-normalized.

    Justificativa: ResNet50 pré-treinado em ImageNet fornece representações visuais
    poderosas que servem como proxy para avaliar se o pré-processamento preserva
    características discriminativas relevantes para modelos de visão.
    """
    _check_dependencies()
    import torch
    import torchvision.models as models
    import torch.nn as nn

    device = torch.device(device)
    model = models.resnet50(pretrained=True)
    # remove a última fc
    model.fc = nn.Identity()
    model.eval()
    model.to(device)

    def extractor(images: List[np.ndarray]) -> np.ndarray:
        """
        Recebe uma lista de imagens BGR (uint8) e retorna embeddings (N, D).
        """
        from torchvision import transforms

        prep = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        tensors = [prep(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in images]
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            feats = model(batch)
            # L2 normalize
            feats = feats / feats.norm(dim=1, keepdim=True)
            return feats.cpu().numpy()

    return extractor


def compute_metrics(originals: List[np.ndarray], processed: List[np.ndarray], device: str = "cpu") -> dict:
    """
    Calcula métricas que quantificam a contribuição do pré-processamento.

    Métricas retornadas:
    - mean_cosine_similarity: média da similaridade coseno entre embeddings (ResNet50)
    - median_cosine_similarity
    - std_cosine_similarity
    - mean_ssim: média do SSIM (estrutura) entre original e processado (convertendo para grayscale)
    - mean_psnr: média do PSNR entre original e processado

    Observação: originals e processed devem ser listas emparelhadas (mesma ordem).
    """
    if len(originals) != len(processed):
        raise ValueError("originals and processed must have the same length")

    n = len(originals)
    if n == 0:
        return {}

    _check_dependencies()
    from skimage.metrics import structural_similarity as ssim

    # Extrair embeddings
    extractor = load_feature_extractor(device)
    feats_orig = extractor(originals)
    feats_proc = extractor(processed)

    # Cosine similarity (since embeddings are L2-normalized, dot product = cosine)
    cosines = (feats_orig * feats_proc).sum(axis=1)

    # SSIM and PSNR
    ssims = []
    psnrs = []
    for o, p in zip(originals, processed):
        # converter para grayscale
        o_gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY) if o.ndim == 3 else o
        p_gray = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY) if p.ndim == 3 else p
        # redimensionar para o mesmo tamanho se necessário
        if o_gray.shape != p_gray.shape:
            p_gray = cv2.resize(p_gray, (o_gray.shape[1], o_gray.shape[0]))

        try:
            s = ssim(o_gray, p_gray, data_range=255)
        except Exception:
            s = float('nan')
        ssims.append(s)

        try:
            ps = cv2.PSNR(o, p)
        except Exception:
            ps = float('nan')
        psnrs.append(ps)

    metrics = {
        "n_frames": n,
        "mean_cosine_similarity": float(np.nanmean(cosines)),
        "median_cosine_similarity": float(np.nanmedian(cosines)),
        "std_cosine_similarity": float(np.nanstd(cosines)),
        "mean_ssim": float(np.nanmean(ssims)),
        "mean_psnr": float(np.nanmean(psnrs)),
    }

    return metrics


def evaluate_and_store(originals: List[np.ndarray], processed: List[np.ndarray], output_dir: str, device: str = "cpu") -> Tuple[dict, str]:
    """
    Executa compute_metrics e salva um JSON com os resultados em output_dir/evaluation.json
    Retorna (metrics, path_to_json).
    """
    os.makedirs(output_dir, exist_ok=True)
    metrics = compute_metrics(originals, processed, device=device)
    out_path = os.path.join(output_dir, "evaluation.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics, out_path
