
"""
Observador EXP-004
Avaliação comparativa das oito configurações com TrackEval.

Execute na raiz:
python -m experiments.exp_004_optimization.evaluate_optimization

Dependência:
pip install trackeval
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

    base = cfg["evaluation"]["exp_003"]
    current = cfg["evaluation"]["exp_004"]
    optimization = cfg["optimization"]["exp_004"]
    dataset = cfg["datasets"]["mot17"]
    tracking = cfg["tracking"]

    return {
        "cfg": cfg,
        "root": root,
        "base": base,
        "current": current,
        "optimization": optimization,
        "dataset": dataset,
        "tracking": tracking,
    }


def load_optimization_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Resumo do EXP-004 não encontrado: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    if summary.get("runs_completed") != len(summary.get("runs", [])):
        raise ValueError(
            "A quantidade de execuções concluídas não corresponde "
            "à quantidade de registros no resumo."
        )

    runs = [
        run for run in summary["runs"]
        if run.get("status") == "completed"
    ]

    if not runs:
        raise ValueError("Nenhuma execução concluída foi encontrada.")

    return {
        **summary,
        "runs": runs,
    }


def prepare_run(
    run: dict[str, Any],
    settings: dict[str, Any],
    gt_rows: list[list[str]],
    seqinfo: Path,
    gt_frames,
    gt_ids,
):
    root = settings["root"]
    base = settings["base"]
    current = settings["current"]
    tracking = settings["tracking"]

    run_name = str(run["run_name"])
    prediction_path = resolve(
        root,
        str(run["prediction_file"])
    )

    limit = int(settings["optimization"]["max_frames"])
    sequence = str(settings["optimization"]["sequence_name"])
    benchmark = str(base.get("benchmark", "MOT17"))

    classes = tracking.get("classes")

    predictions, pred_frames, pred_ids = read_predictions(
        prediction_path,
        limit,
        classes,
    )

    if not predictions:
        raise ValueError(
            f"Nenhuma predição válida encontrada em {run_name}"
        )

    run_output = resolve(
        root,
        current["output_dir"]
    ) / "diagnostics" / run_name

    diagnostic = write_diagnostics(
        run_output,
        limit,
        gt_frames,
        pred_frames,
        gt_ids,
        pred_ids,
    )

    gt_root, tracker_root, output_root, seqmap = prepare_trackeval(
        resolve(root, current["output_dir"]),
        sequence,
        run_name,
        limit,
        gt_rows,
        predictions,
        seqinfo,
        benchmark,
    )

    return {
        "run_name": run_name,
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
    }


def save_json(path: Path, data: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def save_csv(path: Path, results: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "run_name",
        "confidence",
        "image_size",
        "prediction_count",
        "unique_track_ids",
        "prediction_file",
        "diagnostic_file",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )

        writer.writeheader()

        for result in results:
            writer.writerow({
                "run_name": result["run_name"],
                "confidence": result["confidence"],
                "image_size": result["image_size"],
                "prediction_count": result["prediction_count"],
                "unique_track_ids": result["unique_track_ids"],
                "prediction_file": result["prediction_file"],
                "diagnostic_file": result["diagnostic_file"],
            })


def main():
    settings = load_settings()

    root = settings["root"]
    base = settings["base"]
    current = settings["current"]
    optimization = settings["optimization"]
    dataset = settings["dataset"]

    limit = int(optimization["max_frames"])
    sequence = str(base["sequence_name"])
    benchmark = str(base.get("benchmark", "MOT17"))

    output_dir = resolve(
        root,
        current["output_dir"]
    )

    summary_path = resolve(
        root,
        current["summary_file"]
    )

    summary = load_optimization_summary(summary_path)

    sequence_dir = resolve(
        root,
        base["ground_truth_sequence"]
    )

    gt_path = (
        sequence_dir
        / str(dataset["annotation_dir"])
        / str(dataset["annotation_file"])
    )

    seqinfo = sequence_dir / str(dataset["sequence_info"])

    print("\nOBSERVADOR | EXP-004")
    print("AVALIAÇÃO COMPARATIVA TRACK EVAL")
    print("-" * 55)
    print(f"Sequência: {sequence}")
    print(f"Frames: 1–{limit}")
    print(f"Execuções: {len(summary['runs'])}")
    print(f"Benchmark: {benchmark}")

    # Carrega o Ground Truth uma única vez.
    gt_rows, gt_frames, gt_ids = read_gt(
        gt_path,
        limit,
    )

    print(
        f"\nGT válido: {sum(gt_frames.values())}"
        f"\nIDs reais: {len(gt_ids)}"
    )

    prepared_runs = []

    # Prepara os oito rastreadores.
    for index, run in enumerate(summary["runs"], start=1):
        print(
            f"\n[{index}/{len(summary['runs'])}] "
            f"Preparando {run['run_name']}"
        )

        prepared = prepare_run(
            run,
            settings,
            gt_rows,
            seqinfo,
            gt_frames,
            gt_ids,
        )

        prepared_runs.append(prepared)

        print(
            f"Predições: {prepared['prediction_count']} | "
            f"IDs: {prepared['unique_track_ids']}"
        )

    if not prepared_runs:
        raise RuntimeError("Nenhuma execução foi preparada.")

    # Todos os rastreadores compartilham o mesmo GT e sequence map.
    first = prepared_runs[0]

    tracker_names = [
        item["run_name"]
        for item in prepared_runs
    ]

    metrics = [
        str(metric)
        for metric in base.get(
            "metrics",
            ["HOTA", "CLEAR", "Identity"],
        )
    ]

    command = [
        sys.executable,
        str(trackeval_cli()),
        "--GT_FOLDER",
        str(first["gt_root"]),
        "--TRACKERS_FOLDER",
        str(first["tracker_root"]),
        "--TRACKERS_TO_EVAL",
        *tracker_names,
        "--TRACKER_SUB_FOLDER",
        "data",
        "--BENCHMARK",
        benchmark,
        "--SPLIT_TO_EVAL",
        "train",
        "--SEQMAP_FILE",
        str(first["seqmap"]),
        "--DO_PREPROC",
        str(bool(base.get("do_preproc", True))),
        "--METRICS",
        *metrics,
        "--OUTPUT_FOLDER",
        str(first["output_root"]),
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

    log_path = output_dir / "trackeval_execution.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_path.write_text(
        process.stdout + "\n" + process.stderr,
        encoding="utf-8",
    )

    print(process.stdout)

    # Registra os arquivos de saída produzidos pelo TrackEval.
    trackeval_files = []

    for path in first["output_root"].rglob("*"):
        if path.is_file():
            trackeval_files.append(
                str(path.relative_to(output_dir))
            )

    results = []

    for item in prepared_runs:
        diagnostic_path = (
            output_dir
            / "diagnostics"
            / item["run_name"]
            / "diagnostic_summary.json"
        )

        results.append({
            "run_name": item["run_name"],
            "confidence": item["confidence"],
            "image_size": item["image_size"],
            "prediction_count": item["prediction_count"],
            "unique_track_ids": item["unique_track_ids"],
            "prediction_file": item["prediction_file"],
            "diagnostic_file": str(diagnostic_path),
        })

    save_csv(
        output_dir / "optimization_trackeval_runs.csv",
        results,
    )

    final_summary = {
        "experiment": "EXP-004",
        "status": "completed",
        "sequence": sequence,
        "frames_evaluated": limit,
        "benchmark": benchmark,
        "preprocessing_enabled": bool(
            base.get("do_preproc", True)
        ),
        "metrics_requested": metrics,
        "runs_evaluated": len(prepared_runs),
        "runs": results,
        "trackeval_output_dir": str(first["output_root"]),
        "trackeval_log": str(log_path),
        "trackeval_files": trackeval_files,
        "note": (
            "As métricas oficiais devem ser consultadas nos arquivos "
            "gerados pelo TrackEval. As contagens diagnósticas não "
            "representam TP, FP ou FN."
        ),
    }

    save_json(
        output_dir / "optimization_trackeval_summary.json",
        final_summary,
    )

    print("\n" + "=" * 55)
    print("EXP-004 | AVALIAÇÃO CONCLUÍDA")
    print("=" * 55)
    print(f"Execuções avaliadas: {len(prepared_runs)}")
    print(f"Resumo: {output_dir}")
    print(f"Log: {log_path}")
    print(f"Arquivos TrackEval: {first['output_root']}")


if __name__ == "__main__":
    main()
