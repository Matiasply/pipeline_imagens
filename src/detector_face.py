from pathlib import Path
import shutil
import tempfile

import cv2
from detectar_labios import crop_lips

# Usando a Tasks API do MediaPipe (requer 'face_landmarker.task' disponível no ambiente)
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.core import image as mp_image_lib


def resolve_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Não foi possível encontrar o arquivo esperado. "
        "Coloque 'face_landmarker.task' ao lado de 'src/' ou informe um caminho válido."
    )


def stage_ascii_copy(source: Path) -> Path:
    staging_dir = Path(tempfile.gettempdir()) / "pipeline_imagens"
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged_path = staging_dir / source.name
    if not staged_path.exists() or staged_path.stat().st_mtime < source.stat().st_mtime:
        shutil.copy2(source, staged_path)
    return staged_path




script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

model_path = resolve_existing_path(
    script_dir / "face_landmarker.task",
    project_root / "face_landmarker.task",
)
model_path = stage_ascii_copy(model_path)

video_path = resolve_existing_path(
    script_dir / "video.mov",
    project_root / "video.mov",
)
video_path = stage_ascii_copy(video_path)

# Inicializa o Face Landmarker (Tasks API)
base_options = mp_tasks.BaseOptions(model_asset_path=str(model_path))
options = mp_vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)

# Abre o vídeo (mesma entrada do código anterior)
video = cv2.VideoCapture(str(video_path))
frame_idx = 0
fps = video.get(cv2.CAP_PROP_FPS) or 30.0

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Converte BGR -> RGB para entrada do modelo
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Constrói Image compatível com Tasks API
    mp_image = mp_image_lib.Image(mp_image_lib.ImageFormat.SRGB, rgb_frame)

    # Detecta landmarks no modo de vídeo — fornece timestamp em ms
    timestamp_ms = int(frame_idx * (1000.0 / fps))
    frame_idx += 1
    result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

    # A saída pode ter landmarks por face; desenhar landmarks simples como pontos na imagem
    # Estrutura esperada: result.face_landmarks é uma lista (uma entrada por face) contendo landmarks com 'x' e 'y' normalizados
    if hasattr(result, 'face_landmarks') and result.face_landmarks:
        h, w, _ = frame.shape
        for face in result.face_landmarks:
            # face é uma sequence de landmarks
            for lm in face:
                # cada lm tem .x e .y normalizados (0..1)
                x_px = int(lm.x * w)
                y_px = int(lm.y * h)
                cv2.circle(frame, (x_px, y_px), 1, (0, 255, 0), -1)
            # tenta recortar os lábios a partir dos landmarks detectados
            lips_crop_rgb = crop_lips(rgb_frame, face)
            if lips_crop_rgb is not None:
                # OpenCV espera BGR para exibição, converte de RGB -> BGR
                lips_crop_bgr = cv2.cvtColor(lips_crop_rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow("Lips", lips_crop_bgr)

    # Mostra o frame com os pontos desenhados
    #cv2.imshow("Face", frame)

    if cv2.waitKey(30) == 27:
        break

video.release()
cv2.destroyAllWindows()
