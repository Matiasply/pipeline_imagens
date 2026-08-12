from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WordSegment:
    index: int
    start: float
    end: float
    phones: tuple[str, ...]

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def label(self) -> str:
        return "-".join(self.phones).lower()


def _parse_phone(phone_tag: str) -> tuple[str, str | None]:
    if "_" not in phone_tag:
        return phone_tag, None
    phone, boundary = phone_tag.rsplit("_", 1)
    return phone, boundary


def extract_word_segments(alignment_items: list[dict]) -> list[WordSegment]:
    segments: list[WordSegment] = []
    current_start: float | None = None
    current_phones: list[str] = []
    word_index = 1

    for item in alignment_items:
        phone_tag = item["phone"]
        base_phone, boundary = _parse_phone(phone_tag)
        start = float(item["offset"])
        end = start + float(item["duration"])

        if base_phone.startswith("SIL"):
            continue

        if boundary == "B":
            if current_start is not None and current_phones:
                segments.append(
                    WordSegment(
                        index=word_index,
                        start=current_start,
                        end=start,
                        phones=tuple(current_phones),
                    )
                )
                word_index += 1
            current_start = start
            current_phones = [base_phone]
            continue

        if current_start is None:
            current_start = start
            current_phones = [base_phone]
        else:
            current_phones.append(base_phone)

        if boundary == "E":
            segments.append(
                WordSegment(
                    index=word_index,
                    start=current_start,
                    end=end,
                    phones=tuple(current_phones),
                )
            )
            word_index += 1
            current_start = None
            current_phones = []

    if current_start is not None and current_phones:
        last_item = alignment_items[-1]
        last_end = float(last_item["offset"]) + float(last_item["duration"])
        segments.append(
            WordSegment(
                index=word_index,
                start=current_start,
                end=last_end,
                phones=tuple(current_phones),
            )
        )

    return [segment for segment in segments if segment.duration > 0.0]


def cut_video_segment(input_video: Path, output_video: Path, start: float, end: float) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(input_video),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output_video),
    ]
    subprocess.run(command, check=True)


def save_metadata(output_dir: Path, utterance_id: str, segments: list[WordSegment]) -> None:
    metadata = {
        "utterance_id": utterance_id,
        "word_count": len(segments),
        "segments": [
            {
                "index": segment.index,
                "label": segment.label,
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "duration": round(segment.duration, 3),
                "phones": list(segment.phones),
            }
            for segment in segments
        ],
    }
    metadata_path = output_dir / "segments.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def segment_video_by_words(alignment_path: Path, video_path: Path, output_dir: Path) -> list[WordSegment]:
    alignment_data = json.loads(alignment_path.read_text(encoding="utf-8"))
    utterance_id = next(iter(alignment_data))
    alignment_items = alignment_data[utterance_id]
    segments = extract_word_segments(alignment_items)

    output_dir.mkdir(parents=True, exist_ok=True)

    for segment in segments:
        file_name = f"word_{segment.index:02d}_{segment.label}.mp4"
        cut_video_segment(
            input_video=video_path,
            output_video=output_dir / file_name,
            start=segment.start,
            end=segment.end,
        )

    save_metadata(output_dir, utterance_id, segments)
    return segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Separa um vídeo do GRUD Corpus em múltiplos vídeos por palavra."
    )
    parser.add_argument(
        "--alignment",
        required=True,
        type=Path,
        help="Caminho do arquivo JSON de alinhamento fonético (ex: s2_l_bbim3a.json).",
    )
    parser.add_argument(
        "--video",
        required=True,
        type=Path,
        help="Caminho do vídeo correspondente ao alinhamento (ex: s2_l_bbim3a.mov).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Diretório de saída. Padrão: <pasta_do_video>/<nome_do_video>_words",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alignment_path = args.alignment.resolve()
    video_path = args.video.resolve()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = video_path.parent / f"{video_path.stem}_words"
    output_dir = output_dir.resolve()

    if not alignment_path.exists():
        raise FileNotFoundError(f"Arquivo de alinhamento não encontrado: {alignment_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")

    segments = segment_video_by_words(alignment_path, video_path, output_dir)
    print(f"{len(segments)} palavras segmentadas em: {output_dir}")


if __name__ == "__main__":
    main()
