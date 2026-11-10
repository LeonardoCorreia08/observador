import os

# Silencia os logs gigantescos e feios do FFmpeg/OpenCV no terminal
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"

from src.utils.config_loader import (
    load_config,
    get_experiment
)

from src.cameras.camera_manager import CameraManager
from src.detection.detector import ObservadorDetector
from src.validation.training_validator import TrainingValidator
import cv2  # Movido para o topo


def main():
    print("=" * 55)
    print("             OBSERVADOR")
    print("       SISTEMA DE VISÃO COMPUTACIONAL")
    print("=" * 55)

    # Configuração central
    config = load_config()

    # Experimento selecionado
    experiment = get_experiment(
        config,
        "exp_002"
    )

    # Descoberta das câmeras
    cameras = CameraManager(
        max_cameras=10
    )

    available = cameras.discover()

    # Carregamento YOLO
    detector = ObservadorDetector(
        config,
        experiment
    )

    # Verificação do treinamento
    validator = TrainingValidator(
        experiment
    )

    validator.validate()

    # Detecção
    if available:
        camera_id = available[0]["id"]
        
        # Ajuste no CameraManager: Se possível, adicione cv2.CAP_DSHOW lá dentro para Windows
        camera = cameras.open(camera_id)

        print("\n[OBSERVADOR] Iniciando detecção...")
        print("[INFO] Pressione 'q' na janela do vídeo ou 'Ctrl+C' no terminal para sair.")

        max_frames = config.get(
            "tracking", {}
        ).get("max_frames", 300)

        frame_count = 0

        # Adicionado bloco try-except para capturar o Ctrl+C (KeyboardInterrupt) suavemente
        try:
            while frame_count < max_frames:
                ret, frame = camera.read()

                if not ret:
                    print("[ERRO] Falha ao ler frame da câmera.")
                    break

                # O 'verbose=False' silencia o flood de "Results saved to..." no terminal gerado pelo YOLO
                results = detector.detect(frame)

                output = results[0].plot()

                # Garante que a janela fique em primeiro plano no Windows
                cv2.namedWindow("OBSERVADOR - YOLO", cv2.WINDOW_NORMAL)
                cv2.imshow(
                    "OBSERVADOR - YOLO",
                    output
                )

                frame_count += 1

                # Verifica se 'q' ou 'Esc' (27) foi pressionado
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    print("\n[INFO] Encerramento solicitado pelo usuário (tecla pressionada).")
                    break
                
                # Permite fechamento clicando no 'X' da janela (OpenCV no Windows)
                if cv2.getWindowProperty("OBSERVADOR - YOLO", cv2.WND_PROP_VISIBLE) < 1:
                    print("\n[INFO] Janela fechada pelo usuário.")
                    break

        except KeyboardInterrupt:
            print("\n[INFO] Encerramento forçado detectado (Ctrl+C). Fechando com segurança...")
            
        finally:
            # Libera recursos independente de como o loop terminou
            camera.release()
            cv2.destroyAllWindows()

    else:
        print("\n[AVISO] Nenhuma câmera local disponível.")

    print("\n[OBSERVADOR] Execução finalizada.")


if __name__ == "__main__":
    main()