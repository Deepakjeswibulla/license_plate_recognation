from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import os
import uuid
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "mp4"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model = YOLO("license_plate.pt")
CONF_THRESHOLD = 0.5


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    result_media = None
    plates_data = []

    if request.method == "POST":
        file = request.files.get("file")
        if not file or not allowed_file(file.filename):
            return render_template("index.html", error="Invalid file type")

        ext = file.filename.rsplit(".", 1)[1].lower()
        uid = uuid.uuid4().hex
        input_path = os.path.join(UPLOAD_FOLDER, f"{uid}.{ext}")
        output_path = os.path.join(OUTPUT_FOLDER, f"{uid}_out.{ext}")

        file.save(input_path)

        if ext in ["jpg", "jpeg", "png"]:
            image = cv2.imread(input_path)
            results = model(image)[0]

            for box in results.boxes:
                conf = float(box.conf[0])
                if conf < CONF_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                plates_data.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "confidence": round(conf, 2)
                })

            cv2.imwrite(output_path, image)
            result_media = output_path

        elif ext == "mp4":
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (w, h)
            )

            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                results = model(frame)[0]

                for box in results.boxes:
                    conf = float(box.conf[0])
                    if conf < CONF_THRESHOLD:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    plates_data.append({
                        "time": f"frame {frame_count}",
                        "confidence": round(conf, 2)
                    })

                out.write(frame)

            cap.release()
            out.release()
            result_media = output_path

    return render_template(
        "index.html",
        result_media=result_media,
        plates=plates_data
    )


if __name__ == "__main__":
    app.run(debug=True)
