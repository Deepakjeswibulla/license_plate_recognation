from ultralytics import YOLO

class PlateDetector:
    def __init__(self):
        self.model = YOLO("models/license_plate.pt")

    def detect(self, frame):
        plates = []
        results = self.model(frame, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plates.append((x1, y1, x2, y2))
        return plates
