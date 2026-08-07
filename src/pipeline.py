from pathlib import Path

import cv2
from detector_face import (
    resolve_existing_path,
    stage_ascii_copy,
    inicializar_face_landmarker,
    detectar_face,
)
from detectar_labios import crop_lips


def main():
    """Pipeline principal de detecção de face e processamento de lábios."""
    # Configuração de caminhos
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

    output_dir = project_root / "dataset" / "video001"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Inicializa o Face Landmarker
    face_landmarker = inicializar_face_landmarker(model_path)

    # Abre o vídeo
    video = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    fps = video.get(cv2.CAP_PROP_FPS) or 30.0

    # Loop principal de processamento
    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Converte BGR -> RGB para entrada do modelo
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detecta landmarks faciais
        timestamp_ms = int(frame_idx * (1000.0 / fps))
        result = detectar_face(face_landmarker, rgb_frame, timestamp_ms)

        # Processa e exibe landmarks
        if result:
            h, w, _ = frame.shape
            for face in result.face_landmarks:
                # Desenha os landmarks como pontos
                for lm in face:
                    x_px = int(lm.x * w)
                    y_px = int(lm.y * h)
                    cv2.circle(frame, (x_px, y_px), 1, (0, 255, 0), -1)

                # Recorta os lábios
                lips_crop_rgb = crop_lips(rgb_frame, face)
                if lips_crop_rgb is not None:
                    lips_crop_bgr = cv2.cvtColor(lips_crop_rgb, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Lips", lips_crop_bgr)

        # Salva o recorte dos lábios na pasta de saída
        lips_output = None
        if result:
            for face in result.face_landmarks:
                lips_crop_rgb = crop_lips(rgb_frame, face)
                if lips_crop_rgb is not None:
                    lips_crop_bgr = cv2.cvtColor(lips_crop_rgb, cv2.COLOR_RGB2BGR)
                    lips_output = lips_crop_bgr
                    break

        if lips_output is not None:
            frame_filename = output_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(frame_filename), lips_output)

        frame_idx += 1

        if cv2.waitKey(30) == 27:
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()