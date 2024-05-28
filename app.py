from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from PIL import Image
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
import io
from io import BytesIO

INPUT_WIDTH =  640
INPUT_HEIGHT = 640

app = Flask(__name__)

# LOAD YOLO MODEL
net = cv2.dnn.readNetFromONNX('best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Your Azure Computer Vision endpoint and key
endpoint = 'https://ocr-anpr.cognitiveservices.azure.com/'
key = '96566ed7a377482e8d6b1b41da264537'

# Confidence threshold for OCR
confidence_threshold = 0.4

# Create an Image Analysis client
client = ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

def get_detections(img, net):
    image = img.copy()
    row, col, d = image.shape

    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image

    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]

    return input_image, detections

def non_maximum_supression(input_image, detections):
    boxes = []
    confidences = []

    image_w, image_h = input_image.shape[:2]
    x_factor = image_w / INPUT_WIDTH
    y_factor = image_h / INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4]
        if confidence > 0.4:
            class_score = row[5]
            if class_score > 0.25:
                cx, cy, w, h = row[0:4]

                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])

                confidences.append(confidence)
                boxes.append(box)

    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()

    index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45)

    return boxes_np, confidences_np, index

import io as conv
from PIL import Image
from skimage import io as op
import cv2
import numpy as np
from PIL import Image
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

# Your computer vision endpoint and key
endpoint = 'https://ocr-anpr.cognitiveservices.azure.com/'
key = '96566ed7a377482e8d6b1b41da264537'

# Confidence threshold (adjust based on your image quality)
confidence_threshold = 0.4

# Create an Image Analysis client
client = ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

def perform_ocr(image, bbox):
    # Extract the ROI from the image
    x, y, w, h = bbox
    roi = image[y:y+h, x:x+w]
    # Convert the PIL Image to a byte stream
    im_resize = Image.fromarray(roi )
    # Convert the resized image to bytes
    buf = conv.BytesIO()
    im_resize.save(buf, format='JPEG')
    byte_im = buf.getvalue()
    # Extract text (OCR) from the ROI stream
    result = client.analyze(
        image_data=byte_im,
        visual_features=[VisualFeatures.READ]
    )

    # Process OCR results
    potential_plate_numbers = []
    for line in result.read.blocks[0].lines:
        for word in line.words:
            if word.confidence >= confidence_threshold and "IND" not in word.text:
              #if word.text.isalnum():
                potential_plate_numbers.append(word.text)
    if potential_plate_numbers:
      print(f"License plate number (based on confidence): {potential_plate_numbers}")
    else:
      print("No license plate number detected.")

    return potential_plate_numbers

def yolo_predictions(img, net):
    input_image, detections = get_detections(img, net)
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
    text = []

    for ind in index:
        x, y, w, h = boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'plate: {:.0f}%'.format(bb_conf * 100)
        license_text = perform_ocr(img, boxes_np[ind])

        if license_text is not None:
            for detection in license_text:
                text.append(detection)
    print(text)
    return text

@app.route('/detect_number_plate', methods=['POST'])
def detect_number_plate_api():
    if 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400
    
    # Get the image data from the request
    base64_image = request.json['image']
    
    decoded_image = base64.b64decode(base64_image)
    image_bytes = BytesIO(decoded_image)
    # test
    from skimage import io as op
    img = op.imread(image_bytes)
    results = yolo_predictions(img,net)
    
    return jsonify({'number_plates': results}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)