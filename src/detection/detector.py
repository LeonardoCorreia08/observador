from pathlib import Path
from ultralytics import YOLO

from src.utils.config_loader import (
    resolve_path,
    get_device
)


class ObservadorDetector:

    def __init__(self, config, experiment):

        self.config = config
        self.settings = config.get("detection", {})
        self.device = get_device(config)
        self.output_dir = experiment["output_path"]

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        model_path = resolve_path(
            self.settings["model"]
        )

        print("[OBSERVADOR] Carregando YOLO...")
        print(f"[MODELO] {model_path}")

        self.model = YOLO(str(model_path))

        print("[OK] Modelo carregado.")
        self.model.info()


    def detect(self, source):

        results = self.model.predict(
            source=source,
            conf=self.settings.get("confidence", 0.25),
            imgsz=self.settings.get("image_size", 640),
            device=self.device,
            max_det=300,
            
            # Forçando explicitamente para False para silenciar a inferência
            verbose=False, 
            
            save=self.settings.get("save_images", True),
            project=str(self.output_dir),
            name="detection",
            exist_ok=True
        )

        return results