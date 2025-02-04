from flask import Flask, request, jsonify, session
from ultralytics import YOLO
import cv2
import numpy as np
import pytesseract
from PIL import Image
import io

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Required for session storage

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Smallest YOLOv8 model for speed

def analyze_image(image_file):
    """Extract text from image using Tesseract OCR"""
    image = Image.open(image_file)
    return pytesseract.image_to_string(image)

# def detect_objects(image_file):
#     """Detect objects in image using YOLOv8"""
#     image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
#     results = model(image)  # Run YOLO object detection

#     detected_objects = []
#     for result in results:
#         for box in result.boxes:
#             detected_objects.append(result.names[int(box.cls[0])])  # Get class name

#     return list(set(detected_objects)) if detected_objects else ["No objects detected"]


def image_detect_objects(image_file):
    """Detect objects in an image using YOLOv8"""
    image_file.seek(0)  # Reset file pointer to beginning
    image_bytes = np.frombuffer(image_file.read(), np.uint8)
    
    if image_bytes.size == 0:
        return ["Error: Empty image file"]

    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    results = model(image)  # Run YOLO object detection

    detected_objects = []
    for result in results:
        for box in result.boxes:
            detected_objects.append(result.names[int(box.cls[0])])  # Get class name

    return list(set(detected_objects)) if detected_objects else ["No objects detected"]


@app.route("/UploadData", methods=["POST"])
def UploadData():
    text_input = request.data.decode('utf-8')  # Text from request
    uploaded_file = request.files.get('file')  # Uploaded image file
    
    response = {"text_input": text_input, "extracted_text": "", "detected_objects": []}

    # If there is text input, process it
    if text_input.strip():
        response["extracted_text"] = text_input  # Direct text input

    # If an image is uploaded, extract text and detect objects
    if uploaded_file:
        file_ext = uploaded_file.filename.split('.')[-1].lower()
        
        if file_ext in ["jpg", "jpeg", "png", "webp"]:  # Handle image files
            extracted_text = analyze_image(uploaded_file)
            detected_objects = image_detect_objects(uploaded_file)

            response["extracted_text"] = extracted_text.strip()
            response["detected_objects"] = detected_objects

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
