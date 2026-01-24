import streamlit as st
import cv2
import time
from ultralytics import YOLO
import easyocr

# ---------------- CONFIG ----------------
YOLO_CONF_THRESHOLD = 0.6
OCR_CONF_THRESHOLD = 0.8
PLATE_COOLDOWN_SECONDS = 5

st.set_page_config(page_title="License Plate Recognition", layout="wide")
st.title("🚗 License Plate Detection & Recognition")

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    model = YOLO("license_plate.pt")
    reader = easyocr.Reader(['en'], gpu=False)
    return model, reader

plate_model, ocr_reader = load_models()

# ---------------- UTILITIES ----------------
def normalize_plate(text):
    text = text.upper()
    text = ''.join(c for c in text if c.isalnum())
    return text

def preprocess_plate(img):
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    return gray

def detect_and_read(frame):
    results = plate_model(frame)[0]
    detections = []

    for box in results.boxes:
        yolo_conf = float(box.conf[0])
        if yolo_conf < YOLO_CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate = frame[y1:y2, x1:x2]
        if plate.size == 0:
            continue

        proc = preprocess_plate(plate)
        ocr = ocr_reader.readtext(
            proc,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            detail=1
        )

        if not ocr:
            continue

        text, conf = max([(r[1], r[2]) for r in ocr], key=lambda x: x[1])
        if conf < OCR_CONF_THRESHOLD:
            continue

        detections.append({
            "bbox": (x1, y1, x2, y2),
            "text": normalize_plate(text)
        })

    return detections

# ---------------- UI ----------------
mode = st.radio("Select Input", ["Upload Video", "Webcam"])
frame_box = st.empty()
table_box = st.empty()

detected_plates = []
plate_last_seen = {}

# ---------------- VIDEO MODE ----------------
if mode == "Upload Video":
    file = st.file_uploader("Upload video", type=["mp4","avi","mov"])

    if file:
        with open("temp.mp4", "wb") as f:
            f.write(file.read())

        cap = cv2.VideoCapture("temp.mp4")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            detections = detect_and_read(frame)

            for d in detections:
                plate = d["text"]
                last = plate_last_seen.get(plate, 0)

                if now - last >= PLATE_COOLDOWN_SECONDS:
                    detected_plates.append(plate)
                    plate_last_seen[plate] = now

                x1,y1,x2,y2 = d["bbox"]
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,plate,(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

            frame_box.image(frame, channels="BGR")
            table_box.table({"Recognized Plates": list(dict.fromkeys(detected_plates))})

        cap.release()

# ---------------- WEBCAM MODE ----------------
if mode == "Webcam":
    start = st.checkbox("Start Webcam")
    cap = cv2.VideoCapture(0)

    while start:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        detections = detect_and_read(frame)

        for d in detections:
            plate = d["text"]
            last = plate_last_seen.get(plate, 0)

            if now - last >= PLATE_COOLDOWN_SECONDS:
                detected_plates.append(plate)
                plate_last_seen[plate] = now

            x1,y1,x2,y2 = d["bbox"]
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,plate,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

        frame_box.image(frame, channels="BGR")
        table_box.table({"Recognized Plates": list(dict.fromkeys(detected_plates))})

    cap.release()