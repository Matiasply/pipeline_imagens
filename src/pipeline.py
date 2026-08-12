import argparse
import json
import re
from pathlib import Path
from typing import Sequence

import cv2

from detector_face import (
    detectar_face,
    inicializar_face_landmarker,
    resolve_existing_path,
    stage_ascii_copy,
)
from detectar_labios import crop_lips
from lipnet import load_lipnet_model, predict_text_from_video
from preprocessamento import preprocessar

# ==============================================================================
# DICIONÁRIOS DE DECODIFICAÇÃO DO GRID CORPUS
# ==============================================================================
GRID_COMMANDS = {"b": "BIN", "l": "LAY", "p": "PLACE", "s": "SET"}
GRID_COLORS = {"b": "BLUE", "g": "GREEN", "r": "RED", "w": "WHITE"}
GRID_PREPOSITIONS = {"a": "AT", "b": "BY", "i": "IN", "w": "WITH"}
GRID_DIGITS = {
    "0": "ZERO", "1": "ONE", "2": "TWO", "3": "THREE", "4": "FOUR",
    "5": "FIVE", "6": "SIX", "7": "SEVEN", "8": "EIGHT", "9": "NINE",
    "z": "ZERO"
}
GRID_ADVERBS = {"a": "AGAIN", "n": "NOW", "p": "PLEASE", "s": "SOON"}


def decode_grid_utterance(code: str) -> str:
    """
    Decodifica códigos de 6 caracteres do GRID (ex: 'bbat9p') 
    na frase correspondente de 6 palavras (ex: 'BIN BLUE AT T NINE PLEASE').
    """
    code = code.lower().strip()
    if len(code) != 6:
        return code.upper()

    cmd = GRID_COMMANDS.get(code[0], code[0].upper())
    color = GRID_COLORS.get(code[1], code[1].upper())
    prep = GRID_PREPOSITIONS.get(code[2], code[2].upper())
    letter = code[3].upper()
    digit = GRID_DIGITS.get(code[4], code[4].upper())
    adv = GRID_ADVERBS.get(code[5], code[5].upper())

    return f"{cmd} {color} {prep} {letter} {digit} {adv}"

def normalize_sentence_text(value: str) -> str:
    text = str(value).strip().upper()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^A-Z0-9 ]+", "", text)
    return " ".join(text.split())


def levenshtein_distance(source: Sequence[str], target: Sequence[str]) -> int:
    if source == target:
        return 0
    if not source:
        return len(target)
    if not target:
        return len(source)

    previous = list(range(len(target) + 1))
    for i, src_token in enumerate(source, start=1):
        current = [i]
        for j, tgt_token in enumerate(target, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (src_token != tgt_token)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def compute_wer_cer(reference: str, hypothesis: str) -> dict:
    ref_text = normalize_sentence_text(reference)
    hyp_text = normalize_sentence_text(hypothesis)

    ref_tokens = ref_text.split() if ref_text else []
    hyp_tokens = hyp_text.split() if hyp_text else []
    word_distance = levenshtein_distance(ref_tokens, hyp_tokens)
    wer = word_distance / max(len(ref_tokens), 1) if ref_tokens else (0.0 if not hyp_tokens else 1.0)

    ref_chars = list(ref_text)
    hyp_chars = list(hyp_text)
    char_distance = levenshtein_distance(ref_chars, hyp_chars)
    cer = char_distance / max(len(ref_chars), 1) if ref_chars else (0.0 if not hyp_chars else 1.0)

    return {
        "reference": ref_text,
        "hypothesis": hyp_text,
        "distance": char_distance,
        "word_distance": word_distance,
        "wer": float(wer),
        "cer": float(cer),
    }


def sentence_reference_for_video(video_path: Path) -> str:
    stem = video_path.stem
    parts = [part for part in stem.split("_") if part]
    
    code = stem
    if len(parts) >= 3:
        candidate = parts[-1]
        if candidate and candidate.lower() not in {"l", "p"}:
            code = candidate
    elif len(parts) == 1:
        code = parts[0]

    # Transforma 'bbat9p' -> 'BIN BLUE AT T NINE PLEASE'
    return decode_grid_utterance(code)


def _build_face_processed_video(video_path: Path, output_dir: Path, model_path: Path) -> Path:
    """Cria apenas o vídeo pré-processado final, sem salvar imagens frame a frame."""
    output_dir.mkdir(parents=True, exist_ok=True)
    face_landmarker = inicializar_face_landmarker(model_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Vídeo não pôde ser aberto: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    processed_output = output_dir / "processed.mp4"
    writer = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = int(frame_idx * (1000.0 / fps))
        result = detectar_face(face_landmarker, rgb_frame, timestamp_ms)

        lips_output = None
        if result:
            for face in result.face_landmarks:
                lips_crop_rgb = crop_lips(rgb_frame, face)
                if lips_crop_rgb is not None:
                    lips_crop_bgr = cv2.cvtColor(lips_crop_rgb, cv2.COLOR_RGB2BGR)
                    lips_output = preprocessar(lips_crop_bgr)
                    break

        if lips_output is not None:
            if writer is None:
                h_crop, w_crop = lips_output.shape[:2]
                writer = cv2.VideoWriter(
                    str(processed_output),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (w_crop, h_crop),
                )
            if writer is not None and writer.isOpened():
                write_frame = lips_output
                if len(write_frame.shape) == 2:
                    write_frame = cv2.cvtColor(write_frame, cv2.COLOR_GRAY2BGR)
                writer.write(write_frame)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    if not processed_output.exists():
        raise FileNotFoundError(f"Vídeo processado não foi gerado: {processed_output}")

    return processed_output


def evaluate_sentence_against_processed(
    sentence_video: Path,
    processed_video: Path,
    output_dir: Path,
    model=None,
    weights_path: Path | None = None,
) -> dict:
    reference = sentence_reference_for_video(sentence_video)

    if model is None:
        model = load_lipnet_model(weights_path=weights_path, device="cpu")

    original_prediction = predict_text_from_video(sentence_video, model=model, device="cpu")
    processed_prediction = predict_text_from_video(processed_video, model=model, device="cpu")

    original_metrics = compute_wer_cer(reference, original_prediction["predicted_text"])
    processed_metrics = compute_wer_cer(reference, processed_prediction["predicted_text"])

    result = {
        "sentence_video": str(sentence_video),
        "processed_video": str(processed_video),
        "reference": reference,
        "original_prediction": original_prediction["predicted_text"],
        "processed_prediction": processed_prediction["predicted_text"],
        "original_metrics": original_metrics,
        "processed_metrics": processed_metrics,
    }

    summary_path = output_dir / f"{sentence_video.stem}_evaluation.json"
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _find_processed_video_for_sentence(sentence_video: Path, output_dir: Path) -> Path | None:
    candidates = [
        output_dir / "processed.mp4",
        output_dir.parent / "processed.mp4",
        output_dir / sentence_video.stem / "processed.mp4",
        output_dir.parent / sentence_video.stem / "processed.mp4",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def evaluate_processed_outputs(
    sentence_dir: Path,
    processed_root: Path,
    output_dir: Path,
    model=None,
    weights_path: Path | None = None,
) -> list[dict]:
    if not sentence_dir.exists():
        raise FileNotFoundError(f"Diretório de frases não encontrado: {sentence_dir}")

    processed_map: dict[str, Path] = {}
    for processed_video in sorted(processed_root.rglob("processed.mp4")):
        processed_map[processed_video.parent.name] = processed_video

    output_dir.mkdir(parents=True, exist_ok=True)
    result_entries: list[dict] = []
    for sentence_video in sorted(path for path in sentence_dir.rglob("*.mov") if path.is_file()):
        processed_video = processed_map.get(sentence_video.stem)
        if processed_video is None:
            continue

        try:
            evaluation_output_dir = output_dir / sentence_video.stem
            evaluation_output_dir.mkdir(parents=True, exist_ok=True)
            result_entries.append(
                evaluate_sentence_against_processed(
                    sentence_video,
                    processed_video,
                    evaluation_output_dir,
                    model=model,
                    weights_path=weights_path,
                )
            )
        except Exception as exc:  # pragma: no cover - graceful degradation for malformed clips
            result_entries.append(
                {
                    "sentence_video": str(sentence_video),
                    "processed_video": str(processed_video),
                    "status": "skipped",
                    "reason": str(exc),
                }
            )

    summary = {
        "sentence_count": len(result_entries),
        "results": result_entries,
    }
    summary_path = output_dir / "sentence_evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return result_entries


def evaluate_sentences_in_folder(sentence_dir: Path, output_dir: Path, model=None, weights_path: Path | None = None) -> list[dict]:
    if not sentence_dir.exists():
        raise FileNotFoundError(f"Diretório de frases não encontrado: {sentence_dir}")

    return evaluate_processed_outputs(
        sentence_dir,
        output_dir.parent,
        output_dir,
        model=model,
        weights_path=weights_path,
    )


def iter_sentence_videos(sentence_dir: Path) -> list[Path]:
    if sentence_dir.is_file():
        return [sentence_dir]
    if not sentence_dir.exists():
        return []
    return sorted(path for path in sentence_dir.rglob("*.mov") if path.is_file())


def iter_word_videos(words_dir: Path) -> list[Path]:
    if words_dir.is_file():
        return [words_dir]
    if not words_dir.exists():
        return []
    return sorted(path for path in words_dir.rglob("*.mp4") if path.is_file())


def _output_dir_for_video(video_path: Path, root_dir: Path, results_root: Path) -> Path:
    try:
        relative = video_path.relative_to(root_dir)
    except ValueError:
        return results_root / video_path.stem

    return results_root.joinpath(*relative.parts[:-1], video_path.stem)


def run_full_pipeline(
    video_path: Path,
    output_dir: Path,
    sentence_dir: Path | None = None,
    face_model_path: Path | None = None,
    lipnet_model_path: Path | None = None,
) -> Path:
    face_model_path = face_model_path or resolve_existing_path(
        output_dir.parent / "face_landmarker.task",
        Path(__file__).resolve().parent.parent / "face_landmarker.task",
    )
    face_model_path = stage_ascii_copy(face_model_path)

    processed_video = _build_face_processed_video(video_path, output_dir, face_model_path)

    if sentence_dir is not None and sentence_dir.exists():
        evaluation_dir = output_dir / "evaluation"
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        model = load_lipnet_model(weights_path=lipnet_model_path, device="cpu")
        evaluate_sentences_in_folder(sentence_dir, evaluation_dir, model=model, weights_path=lipnet_model_path)

    return processed_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline de frase com LipNet e avaliação WER/CER.")
    parser.add_argument("--video", type=Path, default=None, help="Vídeo original a processar. Se omitido, processa até --limit vídeos em dataset/lombardgrid_front/lombardgrid/front.")
    parser.add_argument("--words-dir", type=Path, default=None, help="Diretório de vídeos de frase/compatibilidade (opcional).")
    parser.add_argument("--model-path", type=Path, default=None, help="Caminho do checkpoint oficial do LipNet (.pt).")
    parser.add_argument("--face-model", type=Path, default=None, help="Caminho do asset MediaPipe FaceLandmarker (.task).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Diretório raiz de saída das transformações em results/.")
    parser.add_argument("--limit", type=int, default=1, help="Máximo de vídeos a processar quando --video não for informado. Use 0 para todos.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    sentence_dir = args.words_dir or project_root / "dataset" / "lombardgrid_front" / "lombardgrid" / "front"
    output_root = args.output_dir or project_root / "results"
    output_root = output_root.resolve()

    video_paths = [args.video] if args.video else iter_sentence_videos(sentence_dir)
    if args.limit and args.limit > 0 and not args.video:
        video_paths = video_paths[: args.limit]
    if not video_paths:
        raise FileNotFoundError(f"Nenhum vídeo encontrado em: {sentence_dir}")

    processed_paths: list[Path] = []
    skipped_entries: list[dict] = []
    for video_path in video_paths:
        try:
            video_path = stage_ascii_copy(video_path)
            if not video_path.exists():
                raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")

            output_dir = output_root if len(video_paths) == 1 else _output_dir_for_video(video_path, sentence_dir, output_root)
            output_dir = output_dir.resolve()

            processed_video = run_full_pipeline(
                video_path,
                output_dir,
                sentence_dir=None,
                face_model_path=args.face_model,
                lipnet_model_path=args.model_path,
            )
            processed_paths.append(processed_video)
            print(f"Vídeo processado salvo em: {processed_video}")
        except Exception as exc:  # pragma: no cover - defensive skip for broken inputs
            skipped_entries.append(
                {
                    "sentence_video": str(video_path),
                    "status": "skipped",
                    "reason": str(exc),
                }
            )
            print(f"Vídeo ignorado por falha: {video_path} -> {exc}")

    if len(processed_paths) == 1:
        evaluation_output_dir = output_root / "evaluation"
        evaluation_output_dir.mkdir(parents=True, exist_ok=True)
        model = load_lipnet_model(weights_path=args.model_path, device="cpu")
        result = evaluate_sentence_against_processed(
            video_paths[0],
            processed_paths[0],
            evaluation_output_dir,
            model=model,
            weights_path=args.model_path,
        )
        summary = {"sentence_count": 1, "results": [result]}
        summary_path = evaluation_output_dir / "sentence_evaluation_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Avaliação concluída: 1 exemplo processado em {evaluation_output_dir}")
        print(f"Resumo final salvo em: {summary_path}")
        print(f"Resultado final: {processed_paths[0]}")
    elif not args.video and sentence_dir.exists():
        evaluation_output_dir = output_root / "evaluation"
        model = load_lipnet_model(weights_path=args.model_path, device="cpu")
        results = evaluate_processed_outputs(sentence_dir, output_root, evaluation_output_dir, model=model, weights_path=args.model_path)
        if skipped_entries:
            results.extend(skipped_entries)
        summary = {"sentence_count": len(results), "results": results}
        summary_path = evaluation_output_dir / "sentence_evaluation_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Avaliação concluída: {len(results)} exemplos processados em {evaluation_output_dir}")
        print(f"Resumo final salvo em: {summary_path}")


if __name__ == "__main__":
    main()