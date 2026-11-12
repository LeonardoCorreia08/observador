"""
Observador EXP-005A
Otimização isolada do parâmetro track_buffer do ByteTrack.

Execute na raiz do projeto:
    python -m experiments.exp_005_optimization.run_optimization

O arquivo bytetrack.yaml original é localizado automaticamente na instalação
do Ultralytics. Uma cópia independente é criada para cada execução.
"""
from __future__ import annotations

import configparser
import copy
import csv
import json
import time
from pathlib import Path
from typing import Any

import cv2
import torch
import ultralytics
import yaml
from ultralytics import YOLO

from src.utils.config_loader import load_config


def project_root(cfg: dict[str, Any]) -> Path:
    package_root = Path(__file__).resolve().parents[2]
    configured = Path(cfg["project"].get("root", "."))
    return configured.resolve() if configured.is_absolute() else (package_root / configured).resolve()


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def natural_frame_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name.lower()
    except ValueError:
        return (10**18, path.name.lower())


def read_sequence(cfg: dict[str, Any], root: Path):
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


def select_device(runtime: dict[str, Any]):
    configured = str(runtime.get("device", "auto")).strip()

    if configured.lower() == "auto":
        if torch.cuda.is_available():
            return 0, torch.cuda.get_device_name(0)
        return "cpu", "CPU"

    if configured.lower() == "cpu":
        return "cpu", "CPU"

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Foi configurado um dispositivo CUDA, mas torch.cuda.is_available() retornou False."
        )

    return configured, torch.cuda.get_device_name(0)


def locate_tracker_yaml(tracker_name: str, root: Path) -> Path:
    """Localiza o YAML informado no projeto ou na instalação do Ultralytics."""
    candidate = Path(tracker_name)

    candidates = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.extend([
            root / candidate,
            Path(ultralytics.__file__).resolve().parent / "cfg" / "trackers" / candidate.name,
        ])

    for path in candidates:
        if path.is_file():
            return path.resolve()

    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Não foi possível localizar {tracker_name}. Caminhos verificados:\n{searched}"
    )


def create_tracker_config(
    source_yaml: Path,
    output_yaml: Path,
    track_buffer: int,
) -> dict[str, Any]:
    if track_buffer < 1:
        raise ValueError("track_buffer deve ser maior que zero.")

    with source_yaml.open("r", encoding="utf-8") as file:
        tracker_cfg = yaml.safe_load(file)

    if not isinstance(tracker_cfg, dict):
        raise ValueError(f"YAML do rastreador inválido: {source_yaml}")

    tracker_type = str(tracker_cfg.get("tracker_type", "")).lower()
    if tracker_type != "bytetrack":
        raise ValueError(
            f"Este experimento foi projetado para ByteTrack, mas encontrou: {tracker_type!r}"
        )

    original_buffer = tracker_cfg.get("track_buffer")
    if original_buffer is None:
        raise KeyError("O YAML original não contém o parâmetro track_buffer.")

    tracker_cfg = copy.deepcopy(tracker_cfg)
    tracker_cfg["track_buffer"] = int(track_buffer)

    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with output_yaml.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            tracker_cfg,
            file,
            sort_keys=False,
            allow_unicode=True,
        )

    return {
        "tracker_type": tracker_type,
        "original_track_buffer": int(original_buffer),
        "tested_track_buffer": int(track_buffer),
        "parameters": tracker_cfg,
    }


def run_combination(
    *,
    model_path: str,
    tracker_yaml: Path,
    tracker_info: dict[str, Any],
    confidence: float,
    image_size: int,
    classes: list[int],
    frames: list[Path],
    max_frames: int,
    output_dir: Path,
    device: Any,
    device_name: str,
    verbose: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "tracking_results.csv"

    selected_frames = frames[:max_frames]
    model = YOLO(model_path)

    total_inference = 0.0
    total_detections = 0
    unique_ids: set[int] = set()
    processed = 0
    start_all = time.perf_counter()

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "frame", "track_id", "class_id", "class_name", "confidence",
            "x1", "y1", "x2", "y2",
        ])

        for frame_number, image_path in enumerate(selected_frames, start=1):
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"Não foi possível ler o frame: {image_path}")

            inference_start = time.perf_counter()
            result = model.track(
                source=image,
                persist=True,
                tracker=str(tracker_yaml),
                conf=confidence,
                imgsz=image_size,
                device=device,
                classes=classes or None,
                verbose=verbose,
                save=False,
            )[0]
            total_inference += time.perf_counter() - inference_start
            processed += 1

            boxes = result.boxes
            if boxes is not None and len(boxes) > 0 and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().tolist()
                ids = boxes.id.int().cpu().tolist()
                class_ids = boxes.cls.int().cpu().tolist()
                confidences = boxes.conf.cpu().tolist()
                names = result.names

                for coords, track_id, class_id, score in zip(
                    xyxy, ids, class_ids, confidences
                ):
                    x1, y1, x2, y2 = (float(value) for value in coords)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    writer.writerow([
                        frame_number,
                        int(track_id),
                        int(class_id),
                        names.get(int(class_id), str(class_id)),
                        round(float(score), 6),
                        round(x1, 3),
                        round(y1, 3),
                        round(x2, 3),
                        round(y2, 3),
                    ])
                    total_detections += 1
                    unique_ids.add(int(track_id))

            if frame_number % 50 == 0 or frame_number == len(selected_frames):
                print(
                    f"  Frame {frame_number}/{len(selected_frames)} | "
                    f"detecções rastreadas: {total_detections}"
                )

    total_time = time.perf_counter() - start_all
    summary = {
        "status": "completed",
        "experiment": "EXP-005A",
        "model": model_path,
        "tracker": "ByteTrack",
        "tracker_source_yaml": str(tracker_info["source_yaml"]),
        "tracker_generated_yaml": str(tracker_yaml),
        "tracker_parameters": tracker_info["parameters"],
        "track_buffer": tracker_info["tested_track_buffer"],
        "original_track_buffer": tracker_info["original_track_buffer"],
        "confidence": confidence,
        "image_size": image_size,
        "classes": classes,
        "device": str(device),
        "device_name": device_name,
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
        "prediction_file": str(csv_path.resolve()),
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    cfg = load_config()
    root = project_root(cfg)

    exp_cfg = cfg["experiments"]["exp_005"]
    opt_cfg = cfg["optimization"]["exp_005"]
    detection_cfg = cfg["detection"]
    tracking_cfg = cfg["tracking"]
    runtime_cfg = cfg.get("runtime", {})
    dataset_cfg = cfg["datasets"]["mot17"]

    sequence_dir, frames, seqinfo = read_sequence(cfg, root)

    max_frames = int(opt_cfg["max_frames"])
    if max_frames < 1:
        raise ValueError("max_frames deve ser maior que zero.")
    if max_frames > len(frames):
        print(f"Aviso: solicitados {max_frames} frames; disponíveis {len(frames)}.")
        max_frames = len(frames)

    confidence = float(opt_cfg["confidence"])
    image_size = int(opt_cfg["image_size"])
    track_buffer_values = [int(value) for value in opt_cfg["track_buffer_values"]]
    classes = [int(value) for value in opt_cfg.get("classes", tracking_cfg.get("classes", []))]

    if not 0 < confidence < 1:
        raise ValueError("confidence deve estar entre 0 e 1.")
    if image_size < 32:
        raise ValueError("image_size deve ser >= 32.")
    if not track_buffer_values or any(value < 1 for value in track_buffer_values):
        raise ValueError("track_buffer_values deve conter inteiros positivos.")
    if len(set(track_buffer_values)) != len(track_buffer_values):
        raise ValueError("track_buffer_values não pode conter valores duplicados.")

    model_path = str(detection_cfg["model"])
    tracker_name = str(opt_cfg.get("tracker", tracking_cfg["tracker"]))
    source_yaml = locate_tracker_yaml(tracker_name, root)
    device, device_name = select_device(runtime_cfg)
    verbose = bool(runtime_cfg.get("verbose", False))

    base_output = resolve(root, exp_cfg["output_dir"])
    base_output.mkdir(parents=True, exist_ok=True)

    print("\nOBSERVADOR | EXP-005A | BYTE TRACK BUFFER")
    print(f"Sequência: {seqinfo.get('name', sequence_dir.name)}")
    print(f"Frames: {max_frames} | Modelo: {model_path}")
    print(f"Confiança: {confidence} | Resolução: {image_size}")
    print(f"Dispositivo: {device_name}")
    print(f"YAML original localizado: {source_yaml}")
    print(f"Valores de track_buffer: {track_buffer_values}")

    results: list[dict[str, Any]] = []

    for index, buffer_value in enumerate(track_buffer_values, start=1):
        run_name = f"run_{index:02d}_track_buffer_{buffer_value}"
        run_dir = base_output / run_name
        tracker_yaml = run_dir / "tracker_config.yaml"

        tracker_info = create_tracker_config(
            source_yaml=source_yaml,
            output_yaml=tracker_yaml,
            track_buffer=buffer_value,
        )
        tracker_info["source_yaml"] = str(source_yaml)

        print(
            f"\n[{index}/{len(track_buffer_values)}] "
            f"Executando track_buffer={buffer_value}"
        )
        print(
            "Parâmetros mantidos: "
            f"high={tracker_info['parameters'].get('track_high_thresh')}, "
            f"low={tracker_info['parameters'].get('track_low_thresh')}, "
            f"new={tracker_info['parameters'].get('new_track_thresh')}, "
            f"match={tracker_info['parameters'].get('match_thresh')}, "
            f"fuse_score={tracker_info['parameters'].get('fuse_score')}"
        )

        summary = run_combination(
            model_path=model_path,
            tracker_yaml=tracker_yaml,
            tracker_info=tracker_info,
            confidence=confidence,
            image_size=image_size,
            classes=classes,
            frames=frames,
            max_frames=max_frames,
            output_dir=run_dir,
            device=device,
            device_name=device_name,
            verbose=verbose,
        )
        summary["run_name"] = run_name
        results.append(summary)

        print(
            f"Concluído: {summary['total_detections']} detecções | "
            f"{summary['unique_track_ids']} IDs | "
            f"{summary['processing_fps']} FPS"
        )

    consolidated = {
        "experiment": "EXP-005A",
        "status": "completed",
        "sequence": str(seqinfo.get("name", sequence_dir.name)),
        "frames_evaluated": max_frames,
        "model": model_path,
        "tracker": "ByteTrack",
        "confidence": confidence,
        "image_size": image_size,
        "track_buffer_values": track_buffer_values,
        "runs_completed": len(results),
        "runs": results,
        "tracker_source_yaml": str(source_yaml),
        "note": (
            "Somente track_buffer foi alterado entre as execuções. "
            "As métricas HOTA, CLEAR e Identity devem ser obtidas pelo TrackEval."
        ),
    }

    summary_path = base_output / "optimization_summary.json"
    summary_path.write_text(
        json.dumps(consolidated, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    csv_path = base_output / "optimization_summary.csv"
    fields = [
        "run_name", "track_buffer", "confidence", "image_size",
        "frames_processed", "total_detections", "unique_track_ids",
        "inference_fps", "processing_fps", "total_processing_time_seconds",
        "prediction_file", "tracker_generated_yaml",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for item in results:
            writer.writerow({field: item.get(field) for field in fields})

    print("\nEXP-005A concluído.")
    print(f"Resumo JSON: {summary_path}")
    print(f"Resumo CSV: {csv_path}")
    print(f"Execuções: {len(results)}")


if __name__ == "__main__":
    main()
