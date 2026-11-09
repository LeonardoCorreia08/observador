
import csv
import json
import time
import configparser

import cv2
import torch
from ultralytics import YOLO

from src.utils.config_loader import load_config, resolve_path


def select_device(runtime_config):
    """Seleciona automaticamente GPU CUDA ou CPU."""

    configured_device = runtime_config.get("device", "auto")

    if configured_device == "auto":
        if torch.cuda.is_available():
            device = 0
            device_name = torch.cuda.get_device_name(0)
        else:
            device = "cpu"
            device_name = "CPU"
    elif configured_device == "cpu":
        device = "cpu"
        device_name = "CPU"
    else:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU CUDA foi configurada, mas não está disponível."
            )

        device = configured_device
        device_name = torch.cuda.get_device_name(0)

    return device, device_name


def load_sequence_info(sequence_dir, info_filename):
    """Lê os metadados da sequência MOT17."""

    info_file = sequence_dir / info_filename

    if not info_file.exists():
        raise FileNotFoundError(
            f"Arquivo de informações não encontrado: {info_file}"
        )

    parser = configparser.ConfigParser()
    parser.read(info_file, encoding="utf-8")

    if not parser.has_section("Sequence"):
        raise ValueError(
            f"Seção [Sequence] não encontrada em {info_file}"
        )

    sequence = parser["Sequence"]

    return {
        "name": sequence.get("name", sequence_dir.name),
        "frame_rate": sequence.getfloat("frameRate", fallback=30.0),
        "seq_length": sequence.getint("seqLength"),
        "im_width": sequence.getint("imWidth"),
        "im_height": sequence.getint("imHeight"),
        "im_ext": sequence.get("imExt", ".jpg"),
    }


def load_ground_truth(gt_file):
    """Carrega e contabiliza as anotações MOT17."""

    if not gt_file.exists():
        raise FileNotFoundError(
            f"Arquivo ground truth não encontrado: {gt_file}"
        )

    rows = []

    with open(gt_file, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            values = line.split(",")

            if len(values) < 6:
                raise ValueError(
                    f"Anotação inválida na linha {line_number}: {line}"
                )

            rows.append({
                "frame": int(float(values[0])),
                "id": int(float(values[1])),
                "x": float(values[2]),
                "y": float(values[3]),
                "width": float(values[4]),
                "height": float(values[5]),
            })

    return rows


def main():
    # ---------------------------------------------------------
    # 1. CONFIGURAÇÕES CENTRAIS
    # ---------------------------------------------------------

    config = load_config()

    project_config = config["project"]
    dataset_config = config["datasets"]["mot17"]
    experiment_config = config["experiments"]["exp_003"]
    detection_config = config["detection"]
    tracking_config = config["tracking"]
    runtime_config = config["runtime"]

    sequence_dir = (
        resolve_path(dataset_config["root"])
        / dataset_config["sequence"]
    )

    image_dir = sequence_dir / dataset_config["image_dir"]

    gt_file = (
        sequence_dir
        / dataset_config["annotation_dir"]
        / dataset_config["annotation_file"]
    )

    output_dir = resolve_path(experiment_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 2. VALIDAÇÃO DOS ARQUIVOS
    # ---------------------------------------------------------

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Diretório de imagens não encontrado: {image_dir}"
        )

    sequence_info = load_sequence_info(
        sequence_dir,
        dataset_config["sequence_info"]
    )

    ground_truth = load_ground_truth(gt_file)

    image_files = sorted(
        image_dir.glob(f"*{sequence_info['im_ext']}")
    )

    if not image_files:
        raise RuntimeError(
            f"Nenhuma imagem encontrada em: {image_dir}"
        )

    # ---------------------------------------------------------
    # 3. PARÂMETROS DE EXECUÇÃO
    # ---------------------------------------------------------

    max_frames = tracking_config.get("max_frames", 300)
    frame_stride = tracking_config.get("frame_stride", 1)

    if max_frames <= 0 or frame_stride <= 0:
        raise ValueError(
            "max_frames e frame_stride devem ser maiores que zero."
        )

    selected_images = image_files[::frame_stride]

    if max_frames:
        selected_images = selected_images[:max_frames]

    device, device_name = select_device(runtime_config)

    model_name = detection_config["model"]
    confidence = detection_config["confidence"]
    image_size = detection_config["image_size"]

    tracker_name = tracking_config["tracker"]
    classes = tracking_config.get("classes")

    # ---------------------------------------------------------
    # 4. CARREGAMENTO DO MODELO
    # ---------------------------------------------------------

    print("=" * 60)
    print(f"{project_config['name']} | EXP-003")
    print(f"Experimento: {experiment_config['name']}")
    print(f"Sequência: {sequence_info['name']}")
    print(f"Frames selecionados: {len(selected_images)}")
    print(f"Modelo: {model_name}")
    print(f"Rastreador: {tracker_name}")
    print(f"Dispositivo: {device_name}")
    print("=" * 60)

    print("\nCarregando modelo...")

    model = YOLO(model_name)

    print("Modelo carregado.")

    # ---------------------------------------------------------
    # 5. PREPARAÇÃO DA SAÍDA
    # ---------------------------------------------------------

    video_file = output_dir / "tracking_output.mp4"
    json_file = output_dir / "tracking_results.json"
    csv_file = output_dir / "tracking_results.csv"
    summary_file = output_dir / "summary.json"

    video_writer = None

    tracking_records = []
    tracked_ids = set()
    track_frame_counts = {}

    total_detections = 0
    processed_frames = 0
    inference_time = 0.0
    start_time = time.perf_counter()

    # ---------------------------------------------------------
    # 6. PROCESSAMENTO E RASTREAMENTO
    # ---------------------------------------------------------

    try:
        for index, image_path in enumerate(selected_images, start=1):

            frame = cv2.imread(str(image_path))

            if frame is None:
                print(
                    f"[WARNING] Não foi possível ler: {image_path.name}"
                )
                continue

            frame_number = int(image_path.stem)

            inference_start = time.perf_counter()

            results = model.track(
                source=frame,
                persist=tracking_config.get("persist", True),
                tracker=tracker_name,
                conf=confidence,
                imgsz=image_size,
                device=device,
                classes=classes,
                verbose=runtime_config.get("verbose", False),
                save=False
            )

            frame_inference_time = (
                time.perf_counter() - inference_start
            )

            inference_time += frame_inference_time

            result = results[0]

            boxes = result.boxes
            annotated_frame = result.plot()

            frame_detections = 0

            if boxes is not None and len(boxes) > 0:

                xyxy = boxes.xyxy.cpu().tolist()
                class_ids = boxes.cls.int().cpu().tolist()
                confidences = boxes.conf.cpu().tolist()

                if boxes.id is not None:
                    track_ids = boxes.id.int().cpu().tolist()
                else:
                    track_ids = [None] * len(boxes)

                for bbox, class_id, conf, track_id in zip(
                    xyxy,
                    class_ids,
                    confidences,
                    track_ids
                ):

                    x1, y1, x2, y2 = bbox

                    record = {
                        "frame": frame_number,
                        "track_id": track_id,
                        "class_id": class_id,
                        "class_name": model.names[int(class_id)],
                        "confidence": round(float(conf), 5),
                        "bbox": {
                            "x1": round(float(x1), 2),
                            "y1": round(float(y1), 2),
                            "x2": round(float(x2), 2),
                            "y2": round(float(y2), 2),
                        }
                    }

                    tracking_records.append(record)
                    frame_detections += 1

                    if track_id is not None:
                        track_id = int(track_id)
                        tracked_ids.add(track_id)

                        track_frame_counts[track_id] = (
                            track_frame_counts.get(track_id, 0) + 1
                        )

            total_detections += frame_detections
            processed_frames += 1

            # Inicializa o vídeo usando o primeiro frame válido.
            if video_writer is None:
                height, width = annotated_frame.shape[:2]

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                video_writer = cv2.VideoWriter(
                    str(video_file),
                    fourcc,
                    sequence_info["frame_rate"] / frame_stride,
                    (width, height)
                )

                if not video_writer.isOpened():
                    raise RuntimeError(
                        f"Não foi possível criar o vídeo: {video_file}"
                    )

            video_writer.write(annotated_frame)

            print(
                f"[{index}/{len(selected_images)}] "
                f"Frame: {frame_number} | "
                f"Detecções: {frame_detections} | "
                f"Tempo: {frame_inference_time:.3f}s"
            )

    finally:
        if video_writer is not None:
            video_writer.release()

    # ---------------------------------------------------------
    # 7. MÉTRICAS DO EXPERIMENTO
    # ---------------------------------------------------------

    total_time = time.perf_counter() - start_time

    average_inference_time = (
        inference_time / processed_frames
        if processed_frames else 0
    )

    average_fps = (
        processed_frames / total_time
        if total_time > 0 else 0
    )

    summary = {
        "experiment": "exp_003",
        "name": experiment_config["name"],
        "dataset": dataset_config["name"],
        "sequence": sequence_info["name"],
        "model": model_name,
        "tracker": tracker_name,
        "device": str(device),
        "device_name": device_name,
        "frames_available": len(image_files),
        "frames_processed": processed_frames,
        "frame_stride": frame_stride,
        "ground_truth_annotations": len(ground_truth),
        "total_detections": total_detections,
        "unique_track_ids": len(tracked_ids),
        "average_detections_per_frame": (
            total_detections / processed_frames
            if processed_frames else 0
        ),
        "average_inference_time_seconds": round(
            average_inference_time, 5
        ),
        "total_processing_time_seconds": round(total_time, 5),
        "average_processing_fps": round(average_fps, 3),
        "video_output": str(video_file),
        "evaluation_status": "NOT_EVALUATED"
    }

    # ---------------------------------------------------------
    # 8. SALVAMENTO DOS RESULTADOS
    # ---------------------------------------------------------

    with open(json_file, "w", encoding="utf-8") as file:
        json.dump(
            tracking_records,
            file,
            indent=4,
            ensure_ascii=False
        )

    csv_fields = [
        "frame",
        "track_id",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2"
    ]

    with open(
        csv_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(file, fieldnames=csv_fields)
        writer.writeheader()

        for record in tracking_records:
            bbox = record["bbox"]

            writer.writerow({
                "frame": record["frame"],
                "track_id": record["track_id"],
                "class_id": record["class_id"],
                "class_name": record["class_name"],
                "confidence": record["confidence"],
                "x1": bbox["x1"],
                "y1": bbox["y1"],
                "x2": bbox["x2"],
                "y2": bbox["y2"]
            })

    summary["tracks"] = [
        {
            "track_id": track_id,
            "frames_tracked": frame_count
        }
        for track_id, frame_count in sorted(
            track_frame_counts.items()
        )
    ]

    with open(summary_file, "w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False
        )

    # ---------------------------------------------------------
    # 9. RESULTADO FINAL
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("RESULTADO DO EXPERIMENTO")
    print("=" * 60)

    for key, value in summary.items():
        if key != "tracks":
            print(f"{key}: {value}")

    print("\nArquivos gerados:")
    print(f"Vídeo: {video_file}")
    print(f"JSON: {json_file}")
    print(f"CSV: {csv_file}")
    print(f"Resumo: {summary_file}")

    print("=" * 60)


if __name__ == "__main__":
    main()
