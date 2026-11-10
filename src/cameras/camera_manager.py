
import cv2


class CameraManager:

    def __init__(self, max_cameras=10):
        self.max_cameras = max_cameras
        self.available_cameras = []

    def discover(self):

        self.available_cameras = []

        print("[OBSERVADOR] Procurando câmeras locais...")

        for camera_id in range(self.max_cameras):

            camera = cv2.VideoCapture(camera_id)

            if camera.isOpened():

                ret, frame = camera.read()

                if ret and frame is not None:

                    info = {
                        "id": camera_id,
                        "width": frame.shape[1],
                        "height": frame.shape[0]
                    }

                    self.available_cameras.append(info)

                    print(
                        f"[OK] Câmera {camera_id}: "
                        f"{info['width']}x{info['height']}"
                    )

            camera.release()

        return self.available_cameras

    def open(self, camera_id):

        camera = cv2.VideoCapture(camera_id)

        if not camera.isOpened():
            raise RuntimeError(
                f"Não foi possível abrir a câmera {camera_id}"
            )

        return camera
