from pathlib import Path
import shutil
import tempfile

from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.core import image as mp_image_lib


def resolve_existing_path(*candidates: Path) -> Path:
    """Resolve o caminho do arquivo, procurando em múltiplos locais."""
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Não foi possível encontrar o arquivo esperado. "
        "Coloque 'face_landmarker.task' ao lado de 'src/' ou informe um caminho válido."
    )


def stage_ascii_copy(source: Path) -> Path:
    """Copia arquivo para diretório temporário para evitar problemas de caminho."""
    staging_dir = Path(tempfile.gettempdir()) / "pipeline_imagens"
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged_path = staging_dir / source.name
    if not staged_path.exists() or staged_path.stat().st_mtime < source.stat().st_mtime:
        shutil.copy2(source, staged_path)
    return staged_path


def inicializar_face_landmarker(model_path: Path) -> mp_vision.FaceLandmarker:
    """Inicializa e retorna o Face Landmarker com as configurações padrão."""
    base_options = mp_tasks.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def detectar_face(face_landmarker: mp_vision.FaceLandmarker, 
                  rgb_frame, 
                  timestamp_ms: int):
    """Detecta landmarks faciais no frame RGB.
    
    Args:
        face_landmarker: Instância do FaceLandmarker
        rgb_frame: Frame em formato RGB
        timestamp_ms: Timestamp do frame em milissegundos
        
    Returns:
        Resultado contendo face_landmarks ou None se não houver faces detectadas
    """
    mp_image = mp_image_lib.Image(mp_image_lib.ImageFormat.SRGB, rgb_frame)
    result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
    return result if hasattr(result, 'face_landmarks') and result.face_landmarks else None