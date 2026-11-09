"""Executa detecção em uma imagem, vídeo ou diretório."""
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--source", required=True, help="Imagem, vídeo ou diretório")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device or None,
        save=True,
        project="runs/trafficvision",
        name="predictions",
    )


if __name__ == "__main__":
    main()
