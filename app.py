import os
import threading
import cv2
from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO('best.pt')

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
    detected_classes = [model.names[int(cls)] for cls in results.boxes.cls] if results.boxes else []

    # Annotate image
    result_img = results.plot()
    result_filename = "result_" + filename
    result_path = os.path.join(UPLOAD_FOLDER, result_filename)
    cv2.imwrite(result_path, result_img)

    # Pass result image path and detected names to template
    relative_result_path = os.path.join('uploads', result_filename)
    return render_template('result.html', result_img=relative_result_path, detected_classes=detected_classes)

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

def run_webcam_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        annotated_frame = results.plot()

        cv2.imshow("YOLOv8 Wild Animal Detector - Press 'q' to Quit", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    threading.Thread(target=run_webcam_detection).start()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
