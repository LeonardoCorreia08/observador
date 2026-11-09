"""Fine-tuning de um detector YOLO com pesos pré-treinados."""
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Caminho do YAML do dataset")
    parser.add_argument("--weights", default="yolo11n.pt", help="Pesos pré-treinados")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="", help="Ex.: 0 para GPU, cpu para CPU")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Carrega pesos pré-treinados: o treinamento abaixo realiza o fine-tuning.
    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        seed=args.seed,
        deterministic=True,
        project="runs/trafficvision",
        name="transfer_learning",
    )


if __name__ == "__main__":
    main()
