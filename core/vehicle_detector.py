from ultralytics import YOLO

class VehicleDetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.allowed_classes = [2, 3, 5, 7]  # car, bike, bus, truck

    def detect(self, frame):
        vehicles = []
        results = self.model(frame, verbose=False)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls in self.allowed_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    vehicles.append((x1, y1, x2, y2))
        return vehicles
