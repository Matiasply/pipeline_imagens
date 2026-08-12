from pathlib import Path

import cv2
from detector_face import (
    resolve_existing_path,
    stage_ascii_copy,
    inicializar_face_landmarker,
    detectar_face,
)
from detectar_labios import crop_lips
from preprocessamento import preprocessar


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

    output_dir = project_root / "result" / "video001"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Inicializa o Face Landmarker
    face_landmarker = inicializar_face_landmarker(model_path)

    # Abre o vídeo
    video = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    fps = video.get(cv2.CAP_PROP_FPS) or 30.0

    # VideoWriter para salvar vídeo processado (inicializado no primeiro frame processado)
    video_writer = None

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

        # Processa e exibe landmarks e recortes de lábios
        lips_output = None
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
                    # Converte para BGR para compatibilidade com o módulo de pré-processamento
                    lips_crop_bgr = cv2.cvtColor(lips_crop_rgb, cv2.COLOR_RGB2BGR)

                    # Aplica pré-processamento (clahe, denoise, suavização, normalização)
                    processed = preprocessar(lips_crop_bgr)

                    # Exibe o resultado processado
                    cv2.imshow("Lips", processed)

                    lips_output = processed
                    break

        # Salva o recorte dos lábios na pasta de saída e escreve no vídeo processado
        if lips_output is not None:
            frame_filename = output_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(frame_filename), lips_output)

            # Inicializa VideoWriter quando tivermos o primeiro recorte
            if video_writer is None:
                h_crop, w_crop = lips_output.shape[:2]
                # VideoWriter espera (width, height)
                video_writer = cv2.VideoWriter(
                    str(output_dir / "processed.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (w_crop, h_crop),
                )

            if video_writer is not None and video_writer.isOpened():
                write_frame = lips_output
                # Se for grayscale, converte para BGR antes de escrever
                if len(write_frame.shape) == 2:
                    write_frame = cv2.cvtColor(write_frame, cv2.COLOR_GRAY2BGR)
                video_writer.write(write_frame)

        frame_idx += 1

        if cv2.waitKey(30) == 27:
            break

    video.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()