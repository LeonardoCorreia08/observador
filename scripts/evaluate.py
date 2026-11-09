"""Avalia um modelo YOLO no split indicado no YAML do dataset."""
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Caminho dos pesos .pt")
    parser.add_argument("--data", required=True, help="YAML do dataset")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        project="runs/trafficvision",
        name=f"evaluation_{args.split}",
    )
    # O Ultralytics também grava artefatos de avaliação na pasta runs.
    print(metrics)


if __name__ == "__main__":
    main()
