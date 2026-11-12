"""Observador EXP-004: varredura controlada de confiança e resolução.

Execute na raiz do projeto:
    python -m experiments.exp_004_detection_optimization.run_optimization

Gera uma pasta independente para cada combinação, com CSV MOT, resumo JSON
 e um resumo consolidado do experimento. Não modifica os resultados do EXP-003.
"""
from __future__ import annotations

import configparser
import csv
import json
import time
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from src.utils.config_loader import load_config


def project_root(cfg: dict[str, Any]) -> Path:
    root = Path(__file__).resolve().parents[2]
    configured = Path(cfg["project"].get("root", "."))
    return configured.resolve() if configured.is_absolute() else (root / configured).resolve()


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def natural_frame_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name.lower()
    except ValueError:
        return (10**18, path.name.lower())


def read_sequence(cfg: dict[str, Any], root: Path) -> tuple[Path, list[Path], dict[str, Any]]:
    dataset = cfg["datasets"]["mot17"]
    sequence_dir = resolve(root, Path(dataset["root"]) / dataset["sequence"])
    image_dir = sequence_dir / str(dataset["image_dir"])
    seqinfo_path = sequence_dir / str(dataset["sequence_info"])
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Diretório de imagens não encontrado: {image_dir}")
    if not seqinfo_path.is_file():
        raise FileNotFoundError(f"Arquivo seqinfo.ini não encontrado: {seqinfo_path}")

    parser = configparser.ConfigParser()
    parser.read(seqinfo_path, encoding="utf-8")
    if not parser.has_section("Sequence"):
        raise ValueError(f"Seção [Sequence] ausente em {seqinfo_path}")
    info = dict(parser["Sequence"])
    extension = info.get("imext", ".jpg")
    frames = sorted(image_dir.glob(f"*{extension}"), key=natural_frame_key)
    if not frames:
        raise FileNotFoundError(f"Nenhuma imagem {extension} encontrada em {image_dir}")
    return sequence_dir, frames, info


def select_device(runtime: dict[str, Any]) -> str | None:
    device = str(runtime.get("device", "auto")).strip()
    return None if device.lower() == "auto" else device


def run_combination(
    *, model_path: str, tracker: str, confidence: float, image_size: int,
    classes: list[int], frames: list[Path], max_frames: int, output_dir: Path,
    device: str | None, verbose: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "tracking_results.csv"
    model = YOLO(model_path)
    selected_frames = frames[:max_frames]
    total_inference = 0.0
    total_detections = 0
    unique_ids: set[int] = set()
    processed = 0
    start_all = time.perf_counter()

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "frame", "track_id", "class_id", "class_name", "confidence",
            "x1", "y1", "x2", "y2"
        ])

        for frame_number, image_path in enumerate(selected_frames, start=1):
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"Não foi possível ler o frame: {image_path}")

            kwargs: dict[str, Any] = {
                "source": image,
                "persist": True,
                "tracker": tracker,
                "conf": confidence,
                "imgsz": image_size,
                "classes": classes or None,
                "verbose": verbose,
            }
            if device is not None:
                kwargs["device"] = device

            inference_start = time.perf_counter()
            result = model.track(**kwargs)[0]
            total_inference += time.perf_counter() - inference_start
            processed += 1

            boxes = result.boxes
            if boxes is None or len(boxes) == 0 or boxes.id is None:
                continue

            xyxy = boxes.xyxy.cpu().tolist()
            ids = boxes.id.int().cpu().tolist()
            class_ids = boxes.cls.int().cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            names = result.names

            for coords, track_id, class_id, score in zip(xyxy, ids, class_ids, confidences):
                x1, y1, x2, y2 = (float(value) for value in coords)
                if x2 <= x1 or y2 <= y1:
                    continue
                writer.writerow([
                    frame_number, int(track_id), int(class_id), names.get(int(class_id), str(class_id)),
                    round(float(score), 6), round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)
                ])
                total_detections += 1
                unique_ids.add(int(track_id))

            if frame_number % 50 == 0 or frame_number == len(selected_frames):
                print(
                    f"  Frame {frame_number}/{len(selected_frames)} | "
                    f"detecções acumuladas: {total_detections}"
                )

    total_time = time.perf_counter() - start_all
    summary = {
        "status": "completed",
        "experiment": "EXP-004",
        "model": model_path,
        "tracker": tracker,
        "confidence": confidence,
        "image_size": image_size,
        "classes": classes,
        "frames_available": len(frames),
        "frames_processed": processed,
        "total_detections": total_detections,
        "unique_track_ids": len(unique_ids),
        "average_detections_per_frame": round(total_detections / processed, 4) if processed else 0.0,
        "total_inference_time_seconds": round(total_inference, 4),
        "average_inference_time_seconds": round(total_inference / processed, 6) if processed else 0.0,
        "inference_fps": round(processed / total_inference, 4) if total_inference else 0.0,
        "total_processing_time_seconds": round(total_time, 4),
        "processing_fps": round(processed / total_time, 4) if total_time else 0.0,
        "prediction_file": str(csv_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    cfg = load_config()
    root = project_root(cfg)
    exp_cfg = cfg["experiments"]["exp_004"]
    opt_cfg = cfg["optimization"]["exp_004"]
    detection_cfg = cfg["detection"]
    tracking_cfg = cfg["tracking"]
    runtime_cfg = cfg.get("runtime", {})

    sequence_dir, frames, seqinfo = read_sequence(cfg, root)
    max_frames = int(opt_cfg.get("max_frames", tracking_cfg.get("max_frames", len(frames))))
    if max_frames < 1:
        raise ValueError("max_frames deve ser maior que zero.")
    if max_frames > len(frames):
        print(f"Aviso: foram solicitados {max_frames} frames, mas existem {len(frames)}.")
        max_frames = len(frames)

    confidence_values = [float(v) for v in opt_cfg["confidence_values"]]
    image_sizes = [int(v) for v in opt_cfg["image_sizes"]]
    if not confidence_values or any(not 0 < v < 1 for v in confidence_values):
        raise ValueError("confidence_values deve conter valores entre 0 e 1.")
    if not image_sizes or any(v < 32 for v in image_sizes):
        raise ValueError("image_sizes deve conter resoluções válidas (>= 32).")

    model_path = str(detection_cfg["model"])
    tracker = str(opt_cfg.get("tracker", tracking_cfg["tracker"]))
    classes = [int(v) for v in opt_cfg.get("classes", tracking_cfg.get("classes", []))]
    device = select_device(runtime_cfg)
    verbose = bool(runtime_cfg.get("verbose", False))
    base_output = resolve(root, exp_cfg["output_dir"])
    base_output.mkdir(parents=True, exist_ok=True)

    print("\nOBSERVADOR | EXP-004 | OTIMIZAÇÃO DA DETECÇÃO")
    print(f"Sequência: {seqinfo.get('name', sequence_dir.name)}")
    print(f"Frames: {max_frames} | Modelo: {model_path} | Rastreador: {tracker}")
    print(f"Confianças: {confidence_values} | Resoluções: {image_sizes}")
    print(f"Dispositivo: {'automático' if device is None else device}")

    results: list[dict[str, Any]] = []
    run_number = 0
    for image_size in image_sizes:
        for confidence in confidence_values:
            run_number += 1
            run_name = f"run_{run_number:02d}_conf_{confidence:.2f}_img_{image_size}".replace(".", "p")
            run_dir = base_output / run_name
            print(
                f"\n[{run_number}/{len(image_sizes) * len(confidence_values)}] "
                f"Confiança={confidence:.2f} | Resolução={image_size}"
            )
            summary = run_combination(
                model_path=model_path,
                tracker=tracker,
                confidence=confidence,
                image_size=image_size,
                classes=classes,
                frames=frames,
                max_frames=max_frames,
                output_dir=run_dir,
                device=device,
                verbose=verbose,
            )
            summary["run_name"] = run_name
            results.append(summary)
            print(
                f"Concluído: {summary['total_detections']} detecções | "
                f"{summary['unique_track_ids']} IDs | {summary['processing_fps']} FPS"
            )

    consolidated = {
        "experiment": "EXP-004",
        "sequence": str(seqinfo.get("name", sequence_dir.name)),
        "frames_evaluated": max_frames,
        "model": model_path,
        "tracker": tracker,
        "confidence_values": confidence_values,
        "image_sizes": image_sizes,
        "runs_completed": len(results),
        "runs": results,
        "note": (
            "Este relatório contém contagens e desempenho operacional. "
            "HOTA, CLEAR e Identity ainda precisam ser calculados pelo TrackEval."
        ),
    }
    summary_path = base_output / "optimization_summary.json"
    summary_path.write_text(
        json.dumps(consolidated, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    csv_summary = base_output / "optimization_summary.csv"
    fields = [
        "run_name", "confidence", "image_size", "frames_processed", "total_detections",
        "unique_track_ids", "average_detections_per_frame", "inference_fps", "processing_fps",
        "total_processing_time_seconds", "prediction_file"
    ]
    with csv_summary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for item in results:
            writer.writerow({field: item.get(field) for field in fields})

    print("\nEXP-004 concluído.")
    print(f"Resumo JSON: {summary_path}")
    print(f"Resumo CSV:  {csv_summary}")
    print("Próxima etapa: avaliar cada arquivo de predição com TrackEval.")


if __name__ == "__main__":
    main()
