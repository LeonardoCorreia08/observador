"""Rastreia veículos com ByteTrack e conta IDs ao cruzar uma linha."""
import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO
from src.trafficvision.counting import LineCounter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True, help="Caminho do vídeo")
    parser.add_argument("--line", type=float, default=0.5,
                        help="Altura relativa da linha (0 a 1)")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--output", default="runs/trafficvision/counting.mp4")
    parser.add_argument("--direction", choices=["down", "up"], default="down")
    args = parser.parse_args()

    if not 0 < args.line < 1:
        raise ValueError("--line deve estar entre 0 e 1.")

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    line_y = int(height * args.line)
    counter = LineCounter(line_y=line_y, direction=args.direction)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Não foi possível criar o vídeo: {output}")

    total = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = model.track(
                frame, persist=True, tracker=args.tracker,
                conf=args.conf, verbose=False
            )[0]

            cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 2)
            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                ids = boxes.id.cpu().numpy().astype(int)
                classes = boxes.cls.cpu().numpy().astype(int)

                for box, track_id, class_id in zip(xyxy, ids, classes):
                    x1, y1, x2, y2 = map(int, box)
                    center_y = (y1 + y2) / 2
                    if counter.update(track_id, center_y):
                        total += 1
                    label = f"{model.names[class_id]} ID:{track_id}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    cv2.putText(frame, label, (x1, max(20, y1 - 7)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)

            cv2.putText(frame, f"Contagem: {total}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    print(f"Contagem final: {total}")
    print(f"Vídeo salvo em: {output}")


if __name__ == "__main__":
    main()
