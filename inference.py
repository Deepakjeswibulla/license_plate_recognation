import cv2
import time
import re
from ultralytics import YOLO
import easyocr

model = YOLO("models/license_plate.pt")
ocr = easyocr.Reader(['en'], gpu=False)

_plate_db = {}

def get_plate_db():
    return _plate_db

def clean_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def detect_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (960, 540))
        results = model(frame, conf=0.4)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            ocr_results = ocr.readtext(crop)
            if not ocr_results:
                continue

            plate = clean_text(ocr_results[0][1])
            if len(plate) < 5:
                continue

            now = time.strftime("%H:%M:%S")

            if plate not in _plate_db:
                _plate_db[plate] = {
                    "count": 1,
                    "first_seen": now,
                    "last_seen": now
                }
            else:
                _plate_db[plate]["count"] += 1
                _plate_db[plate]["last_seen"] = now

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, plate, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        _, jpeg = cv2.imencode(".jpg", frame)
        yield jpeg.tobytes()

    cap.release()
