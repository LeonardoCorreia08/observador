import json
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from src.utils.config_loader import load_config, resolve_path


def select_device(runtime_config):
    """
    Seleciona automaticamente GPU CUDA ou CPU.
    """

    configured_device = runtime_config.get("device", "auto")

    if configured_device == "auto":
        if torch.cuda.is_available():
            device = 0
            device_name = torch.cuda.get_device_name(0)

            print(f"[DEVICE] GPU NVIDIA detectada: {device_name}")

        else:
            device = "cpu"
            device_name = "CPU"

            print("[DEVICE] GPU CUDA indisponível. Utilizando CPU.")

    elif configured_device == "cpu":
        device = "cpu"
        device_name = "CPU"

    else:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Foi solicitada uma GPU, mas CUDA não está disponível."
            )

        device = configured_device
        device_name = torch.cuda.get_device_name(0)

    return device, device_name


def main():

    # --------------------------------------------------
    # 1. CARREGAMENTO DAS CONFIGURAÇÕES
    # --------------------------------------------------

    config = load_config()

    dataset_config = config["dataset"]
    detection_config = config["detection"]
    runtime_config = config["runtime"]

    experiment_config = config["experiments"]["exp_002"]

    dataset_root = resolve_path(dataset_config["root"])

    image_dir = dataset_root / dataset_config["image_dir"]

    output_dir = resolve_path(
        experiment_config["output_dir"]
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    annotated_dir = output_dir / "annotated"

    if detection_config["save_images"]:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 2. VERIFICAÇÃO DOS DADOS
    # --------------------------------------------------

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Diretório de imagens não encontrado: {image_dir}"
        )

    images = sorted(image_dir.glob("*.jpg"))

    if not images:
        raise RuntimeError("Nenhuma imagem encontrada no COCO.")

    max_images = detection_config["max_images"]

    if max_images > 0:
        images = images[:max_images]

    print("=" * 60)
    print("OBSERVADOR | EXP-002")
    print(f"Experimento: {experiment_config['name']}")
    print(f"Imagens selecionadas: {len(images)}")
    print("=" * 60)

    # --------------------------------------------------
    # 3. SELEÇÃO AUTOMÁTICA GPU / CPU
    # --------------------------------------------------

    device, device_name = select_device(runtime_config)

    # --------------------------------------------------
    # 4. CARREGAMENTO DO MODELO
    # --------------------------------------------------

    model_name = detection_config["model"]

    print(f"\nCarregando modelo: {model_name}")

    model = YOLO(model_name)

    print("Modelo carregado.")

    # --------------------------------------------------
    # 5. EXECUÇÃO DA DETECÇÃO
    # --------------------------------------------------

    all_detections = []

    total_objects = 0
    total_inference_time = 0.0

    for index, image_path in enumerate(images, start=1):

        start_time = time.perf_counter()

        results = model.predict(
            source=str(image_path),
            conf=detection_config["confidence"],
            imgsz=detection_config["image_size"],
            device=device,
            verbose=runtime_config["verbose"],
            save=False
        )

        elapsed = time.perf_counter() - start_time

        total_inference_time += elapsed

        result = results[0]

        image_detections = []

        for box in result.boxes:

            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            coordinates = [
                round(float(value), 2)
                for value in box.xyxy[0].tolist()
            ]

            detection = {
                "class_id": class_id,
                "class_name": model.names[class_id],
                "confidence": round(confidence, 4),
                "bbox_xyxy": coordinates
            }

            image_detections.append(detection)

        object_count = len(image_detections)
        total_objects += object_count

        all_detections.append({
            "image": image_path.name,
            "inference_time_seconds": round(elapsed, 4),
            "objects_detected": object_count,
            "detections": image_detections
        })

        # Salva imagem com bounding boxes
        if detection_config["save_images"]:

            annotated_image = result.plot()

            output_image = annotated_dir / image_path.name

            success = cv2.imwrite(
                str(output_image),
                annotated_image
            )

            if not success:
                print(f"[WARNING] Falha ao salvar: {output_image}")

        print(
            f"[{index}/{len(images)}] "
            f"{image_path.name} | "
            f"Objetos: {object_count} | "
            f"Tempo: {elapsed:.3f}s"
        )

    # --------------------------------------------------
    # 6. CÁLCULO DAS MÉTRICAS
    # --------------------------------------------------

    image_count = len(images)

    average_inference = (
        total_inference_time / image_count
        if image_count else 0
    )

    average_objects = (
        total_objects / image_count
        if image_count else 0
    )

    # --------------------------------------------------
    # 7. RELATÓRIO
    # --------------------------------------------------

    summary = {
        "experiment": "exp_002",
        "name": experiment_config["name"],
        "dataset": dataset_config["name"],
        "model": model_name,
        "device": str(device),
        "device_name": device_name,
        "images_processed": image_count,
        "total_objects_detected": total_objects,
        "average_objects_per_image": round(
            average_objects, 2
        ),
        "total_inference_time_seconds": round(
            total_inference_time, 4
        ),
        "average_inference_time_seconds": round(
            average_inference, 4
        ),
        "average_inference_fps": round(
            1 / average_inference, 2
        ) if average_inference > 0 else 0,
        "confidence_threshold": detection_config["confidence"],
        "image_size": detection_config["image_size"]
    }

    summary_file = output_dir / "summary.json"

    detections_file = output_dir / "detections.json"

    with open(summary_file, "w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False
        )

    with open(detections_file, "w", encoding="utf-8") as file:
        json.dump(
            all_detections,
            file,
            indent=4,
            ensure_ascii=False
        )

    # --------------------------------------------------
    # 8. RESULTADO FINAL
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("RESULTADO DO EXPERIMENTO")
    print("=" * 60)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nRelatórios salvos em:")
    print(output_dir)

    print("=" * 60)


if __name__ == "__main__":
    main()