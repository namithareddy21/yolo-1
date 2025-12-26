from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = Flask(__name__)
model = YOLO("yolov8n.pt")

descriptions = {
    "person": "A human being detected in the scene.",
    "car": "A four-wheeled motor vehicle.",
    "bottle": "A container used to store liquids.",
    "cell phone": "A portable communication device.",
    "chair": "A piece of furniture used for sitting."
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    data = request.json["image"]
    img_bytes = base64.b64decode(data.split(",")[1])
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = model(frame)[0]

    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        detections.append({
            "label": label,
            "x": x1,
            "y": y1,
            "w": x2 - x1,
            "h": y2 - y1,
            "desc": descriptions.get(label, "Detected object")
        })

    return jsonify({
        "count": len(detections),
        "detections": detections
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
