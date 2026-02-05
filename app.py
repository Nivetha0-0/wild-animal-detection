import os
import cv2
from flask import Flask, render_template, request
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model
model = YOLO('best.pt')

# Upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('image')
    if not file:
        return "No file uploaded", 400

    filename = file.filename or "uploaded.jpg"
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    # Run YOLO model
    results = model(save_path)[0]
    detected_classes = (
        [model.names[int(cls)] for cls in results.boxes.cls]
        if results.boxes else []
    )

    # Annotate image
    result_img = results.plot()
    result_filename = "result_" + filename
    result_path = os.path.join(UPLOAD_FOLDER, result_filename)
    cv2.imwrite(result_path, result_img)

    # Path used by HTML (relative to static/)
    relative_result_path = os.path.join('uploads', result_filename)

    return render_template(
        'result.html',
        result_img=relative_result_path,
        detected_classes=detected_classes
    )

# -------------------------------
# Webcam code DISABLED (Option A)
# -------------------------------
# @app.route('/webcam')
# def webcam():
#     return render_template('webcam.html')
#
# def run_webcam_detection():
#     pass
#
# @app.route('/start_webcam', methods=['POST'])
# def start_webcam():
#     pass
# -------------------------------

if __name__ == '__main__':
    app.run()
