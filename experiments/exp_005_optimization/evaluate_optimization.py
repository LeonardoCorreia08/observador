"""
Observador EXP-005A
Avaliação das configurações de track_buffer usando TrackEval.

Execute na raiz do projeto:
    python -m experiments.exp_005_optimization.evaluate_optimization

Reutiliza as funções de leitura, diagnóstico e preparação MOT17 do EXP-003.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.utils.config_loader import load_config

from experiments.exp_003_tracking.evaluate_tracking import (
    get_root,
    resolve,
    read_gt,
    read_predictions,
    write_diagnostics,
    prepare_trackeval,
    trackeval_cli,
)


def load_settings():
    cfg = load_config()
    if not isinstance(cfg, dict):
        raise TypeError("A configuração deve ser um dicionário.")

    root = get_root(cfg)
    return {
        "cfg": cfg,
        "root": root,
        "base": cfg["evaluation"]["exp_003"],
        "current": cfg["evaluation"]["exp_005"],
        "optimization": cfg["optimization"]["exp_005"],
        "dataset": cfg["datasets"]["mot17"],
        "tracking": cfg["tracking"],
    }


def load_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Resumo do EXP-005 não encontrado: {path}")

    with path.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    runs = [
        run for run in summary.get("runs", [])
        if run.get("status") == "completed"
    ]
    if summary.get("runs_completed") != len(runs) or not runs:
        raise ValueError("Resumo sem execuções concluídas consistentes.")

    return {**summary, "runs": runs}


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def main():
    settings = load_settings()
    root = settings["root"]
    base = settings["base"]
    current = settings["current"]
    optimization = settings["optimization"]
    dataset = settings["dataset"]
    tracking = settings["tracking"]

    limit = int(optimization["max_frames"])
    sequence = str(optimization["sequence_name"])
    benchmark = str(base.get("benchmark", "MOT17"))

    output_dir = resolve(root, current["output_dir"])
    summary_path = resolve(root, current["summary_file"])
    summary = load_summary(summary_path)

    sequence_dir = resolve(root, base["ground_truth_sequence"])
    gt_path = (
        sequence_dir
        / str(dataset["annotation_dir"])
        / str(dataset["annotation_file"])
    )
    seqinfo = sequence_dir / str(dataset["sequence_info"])

    print("\nOBSERVADOR | EXP-005A")
    print("AVALIAÇÃO COMPARATIVA TRACKEVAL")
    print("-" * 55)
    print(f"Sequência: {sequence}")
    print(f"Frames: 1–{limit}")
    print(f"Execuções: {len(summary['runs'])}")
    print(f"Benchmark: {benchmark}")

    gt_rows, gt_frames, gt_ids = read_gt(gt_path, limit)
    print(
        f"\nGT válido: {sum(gt_frames.values())}"
        f"\nIDs reais: {len(gt_ids)}"
    )

    prepared_runs = []
    for index, run in enumerate(summary["runs"], start=1):
        run_name = str(run["run_name"])
        prediction_path = resolve(root, str(run["prediction_file"]))

        print(f"\n[{index}/{len(summary['runs'])}] Preparando {run_name}")

        predictions, pred_frames, pred_ids = read_predictions(
            prediction_path,
            limit,
            tracking.get("classes"),
        )
        if not predictions:
            raise ValueError(f"Nenhuma predição válida encontrada em {run_name}")

        run_output = output_dir / "diagnostics" / run_name
        diagnostic = write_diagnostics(
            run_output,
            limit,
            gt_frames,
            pred_frames,
            gt_ids,
            pred_ids,
        )

        gt_root, tracker_root, output_root, seqmap = prepare_trackeval(
            output_dir,
            sequence,
            run_name,
            limit,
            gt_rows,
            predictions,
            seqinfo,
            benchmark,
        )

        prepared_runs.append({
            "run_name": run_name,
            "track_buffer": run["track_buffer"],
            "confidence": run["confidence"],
            "image_size": run["image_size"],
            "prediction_file": str(prediction_path),
            "prediction_count": len(predictions),
            "unique_track_ids": len(pred_ids),
            "diagnostic": diagnostic,
            "gt_root": gt_root,
            "tracker_root": tracker_root,
            "output_root": output_root,
            "seqmap": seqmap,
        })
        print(f"Predições: {len(predictions)} | IDs: {len(pred_ids)}")

    first = prepared_runs[0]
    tracker_names = [item["run_name"] for item in prepared_runs]
    metrics = [
        str(metric)
        for metric in base.get("metrics", ["HOTA", "CLEAR", "Identity"])
    ]

    command = [
        sys.executable,
        str(trackeval_cli()),
        "--GT_FOLDER", str(first["gt_root"]),
        "--TRACKERS_FOLDER", str(first["tracker_root"]),
        "--TRACKERS_TO_EVAL", *tracker_names,
        "--TRACKER_SUB_FOLDER", "data",
        "--BENCHMARK", benchmark,
        "--SPLIT_TO_EVAL", "train",
        "--SEQMAP_FILE", str(first["seqmap"]),
        "--DO_PREPROC", str(bool(base.get("do_preproc", True))),
        "--METRICS", *metrics,
        "--OUTPUT_FOLDER", str(first["output_root"]),
    ]

    print("\n" + "-" * 55)
    print("EXECUTANDO TRACKEVAL")
    print(f"Rastreadores: {len(tracker_names)}")
    print(f"Métricas: {', '.join(metrics)}")
    print("-" * 55)

    process = subprocess.run(
        command,
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "trackeval_execution.log"
    log_path.write_text(
        process.stdout + "\n" + process.stderr,
        encoding="utf-8",
    )
    print(process.stdout)

    trackeval_files = [
        str(path.relative_to(output_dir))
        for path in first["output_root"].rglob("*")
        if path.is_file()
    ]

    results = []
    for item in prepared_runs:
        diagnostic_path = (
            output_dir / "diagnostics" / item["run_name"]
            / "diagnostic_summary.json"
        )
        results.append({
            "run_name": item["run_name"],
            "track_buffer": item["track_buffer"],
            "confidence": item["confidence"],
            "image_size": item["image_size"],
            "prediction_count": item["prediction_count"],
            "unique_track_ids": item["unique_track_ids"],
            "prediction_file": item["prediction_file"],
            "diagnostic_file": str(diagnostic_path),
        })

    csv_path = output_dir / "optimization_trackeval_runs.csv"
    columns = [
        "run_name", "track_buffer", "confidence", "image_size",
        "prediction_count", "unique_track_ids",
        "prediction_file", "diagnostic_file",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)

    final_summary = {
        "experiment": "EXP-005A",
        "status": "completed",
        "sequence": sequence,
        "frames_evaluated": limit,
        "benchmark": benchmark,
        "preprocessing_enabled": bool(base.get("do_preproc", True)),
        "metrics_requested": metrics,
        "runs_evaluated": len(prepared_runs),
        "runs": results,
        "trackeval_output_dir": str(first["output_root"]),
        "trackeval_log": str(log_path),
        "trackeval_files": trackeval_files,
        "note": (
            "As métricas oficiais devem ser consultadas nos arquivos "
            "gerados pelo TrackEval. O CSV de execuções contém dados "
            "operacionais e diagnósticos, não métricas oficiais."
        ),
    }
    save_json(output_dir / "optimization_trackeval_summary.json", final_summary)

    print("\n" + "=" * 55)
    print("EXP-005A | AVALIAÇÃO CONCLUÍDA")
    print("=" * 55)
    print(f"Execuções avaliadas: {len(prepared_runs)}")
    print(f"Resumo: {output_dir}")
    print(f"Log: {log_path}")
    print(f"Arquivos TrackEval: {first['output_root']}")


if __name__ == "__main__":
    main()
