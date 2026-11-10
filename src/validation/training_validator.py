
from pathlib import Path
import pandas as pd


class TrainingValidator:

    def __init__(self, experiment):
        self.output_dir = experiment["output_path"]

    def validate(self):

        print("\n[OBSERVADOR] Verificando treinamento...")

        weights_dir = self.output_dir / "weights"

        best_weights = weights_dir / "best.pt"
        last_weights = weights_dir / "last.pt"

        results_file = self.output_dir / "results.csv"

        report = {
            "best_weights": best_weights.exists(),
            "last_weights": last_weights.exists(),
            "results_csv": results_file.exists()
        }

        for name, status in report.items():

            print(
                f"[{'OK' if status else 'AUSENTE'}] {name}"
            )

        if results_file.exists():

            df = pd.read_csv(results_file)

            print("\n===== RELATÓRIO DE TREINAMENTO =====")

            print("Épocas registradas:", len(df))

            metrics = [
                "metrics/precision(B)",
                "metrics/recall(B)",
                "metrics/mAP50(B)",
                "metrics/mAP50-95(B)"
            ]

            for metric in metrics:

                if metric in df.columns:

                    value = df[metric].dropna()

                    if not value.empty:

                        print(
                            f"{metric}: {value.iloc[-1]:.4f}"
                        )

            print("\n[OK] Análise concluída.")

        return report
