import argparse
import json
from pathlib import Path


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values):
    if not values:
        return None
    return sum(values) / len(values)


def build_report(summary: dict) -> dict:
    results = summary.get("results", [])

    paired_entries = []
    skipped_entries = []

    for entry in results:
        if entry.get("status") == "skipped":
            skipped_entries.append(entry)
            continue

        original = entry.get("original_metrics", {})
        processed = entry.get("processed_metrics", {})

        orig_wer = _safe_float(original.get("wer"))
        proc_wer = _safe_float(processed.get("wer"))
        orig_cer = _safe_float(original.get("cer"))
        proc_cer = _safe_float(processed.get("cer"))

        if None in (orig_wer, proc_wer, orig_cer, proc_cer):
            skipped_entries.append(
                {
                    "sentence_video": entry.get("sentence_video"),
                    "status": "skipped",
                    "reason": "missing_metrics",
                }
            )
            continue

        paired_entries.append(
            {
                "sentence_video": entry.get("sentence_video"),
                "delta_wer": proc_wer - orig_wer,
                "delta_cer": proc_cer - orig_cer,
                "original_wer": orig_wer,
                "processed_wer": proc_wer,
                "original_cer": orig_cer,
                "processed_cer": proc_cer,
            }
        )

    original_wers = [item["original_wer"] for item in paired_entries]
    processed_wers = [item["processed_wer"] for item in paired_entries]
    original_cers = [item["original_cer"] for item in paired_entries]
    processed_cers = [item["processed_cer"] for item in paired_entries]

    improved_wer = [item for item in paired_entries if item["delta_wer"] < 0]
    improved_cer = [item for item in paired_entries if item["delta_cer"] < 0]
    tied_wer = [item for item in paired_entries if item["delta_wer"] == 0]
    tied_cer = [item for item in paired_entries if item["delta_cer"] == 0]

    report = {
        "total_entries": len(results),
        "valid_metric_entries": len(paired_entries),
        "skipped_entries": len(skipped_entries),
        "mean_original_wer": _mean(original_wers),
        "mean_processed_wer": _mean(processed_wers),
        "mean_original_cer": _mean(original_cers),
        "mean_processed_cer": _mean(processed_cers),
        "gain_wer": (_mean(original_wers) - _mean(processed_wers)) if paired_entries else None,
        "gain_cer": (_mean(original_cers) - _mean(processed_cers)) if paired_entries else None,
        "improved_wer_count": len(improved_wer),
        "improved_cer_count": len(improved_cer),
        "tied_wer_count": len(tied_wer),
        "tied_cer_count": len(tied_cer),
        "improved_wer_rate": (len(improved_wer) / len(paired_entries)) if paired_entries else None,
        "improved_cer_rate": (len(improved_cer) / len(paired_entries)) if paired_entries else None,
        "top_5_cer_improvements": sorted(paired_entries, key=lambda item: item["delta_cer"])[:5],
        "top_5_wer_improvements": sorted(paired_entries, key=lambda item: item["delta_wer"])[:5],
    }

    return report


def print_report(report: dict) -> None:
    print("=== Evaluation Effectiveness Report ===")
    print(f"Total entries: {report['total_entries']}")
    print(f"Valid metric entries: {report['valid_metric_entries']}")
    print(f"Skipped entries: {report['skipped_entries']}")
    print(f"Mean original WER: {report['mean_original_wer']}")
    print(f"Mean processed WER: {report['mean_processed_wer']}")
    print(f"Mean original CER: {report['mean_original_cer']}")
    print(f"Mean processed CER: {report['mean_processed_cer']}")
    print(f"WER gain (positive is better): {report['gain_wer']}")
    print(f"CER gain (positive is better): {report['gain_cer']}")
    print(f"Improved WER count: {report['improved_wer_count']}")
    print(f"Improved CER count: {report['improved_cer_count']}")
    print(f"Improved WER rate: {report['improved_wer_rate']}")
    print(f"Improved CER rate: {report['improved_cer_rate']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera relatório agregado de eficácia do pré-processamento a partir do summary JSON.")
    parser.add_argument("--input", type=Path, required=True, help="Caminho para sentence_evaluation_summary.json")
    parser.add_argument("--output", type=Path, default=None, help="Caminho opcional para salvar o relatório agregado em JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {args.input}")

    summary = json.loads(args.input.read_text(encoding="utf-8"))
    report = build_report(summary)
    print_report(report)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Relatório salvo em: {args.output}")


if __name__ == "__main__":
    main()