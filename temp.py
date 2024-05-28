#!/usr/bin/env python
# coding: utf-8

# In[1]:


# settings
INPUT_WIDTH =  640
INPUT_HEIGHT = 640


# In[4]:


import os
import string
import cv2
import numpy as np
import plotly.express as px # not needed
from skimage import io
from shutil import copy


# In[6]:


# LOAD YOLO MODEL
net = cv2.dnn.readNetFromONNX('best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# In[7]:


def get_detections(img,net):
    # 1.CONVERT IMAGE TO YOLO FORMAT
    image = img.copy()
    row, col, d = image.shape

    max_rc = max(row,col)
    input_image = np.zeros((max_rc,max_rc,3),dtype=np.uint8)
    input_image[0:row,0:col] = image

    # 2. GET PREDICTION FROM YOLO MODEL
    blob = cv2.dnn.blobFromImage(input_image,1/255,(INPUT_WIDTH,INPUT_HEIGHT),swapRB=True,crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]

    return input_image, detections

def non_maximum_supression(input_image,detections):

    # 3. FILTER DETECTIONS BASED ON CONFIDENCE AND PROBABILIY SCORE

    # center x, center y, w , h, conf, proba
    boxes = []
    confidences = []

    image_w, image_h = input_image.shape[:2]
    x_factor = image_w/INPUT_WIDTH
    y_factor = image_h/INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4] # confidence of detecting license plate
        if confidence > 0.4:
            class_score = row[5] # probability score of license plate
            if class_score > 0.25:
                cx, cy , w, h = row[0:4]

                left = int((cx - 0.5*w)*x_factor)
                top = int((cy-0.5*h)*y_factor)
                width = int(w*x_factor)
                height = int(h*y_factor)
                box = np.array([left,top,width,height])

                confidences.append(confidence)
                boxes.append(box)

    # 4.1 CLEAN
    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()

    # 4.2 NMS
    index = cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.45)

    return boxes_np, confidences_np, index

def drawings(image,boxes_np,confidences_np,index):

    # 5. Drawings
    for ind in index:
        x,y,w,h =  boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'plate: {:.0f}%'.format(bb_conf*100)
        license_text = perform_ocr(image, boxes_np[ind])

    if license_text is not None: # Check if perform_ocr returned an error (None)
          for detection in license_text:
            print(detection)  # Text detected
    else:
        print("Error during OCR for this image")  # Handle the error case


# In[8]:


# predictions flow with return result
def yolo_predictions(img,net):
    # step-1: detections
    input_image, detections = get_detections(img,net)
    # step-2: NMS
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
    # step-3: Drawings
    result_img = drawings(img,boxes_np,confidences_np,index)
    return result_img


# In[10]:


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
    fig = px.imshow(roi)
    fig.show()
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


# In[15]:


import base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_string


# In[16]:


import base64
from io import BytesIO
from skimage import io as op
import cv2
import numpy as np

def process_base64_image(base64_string):
    # Decode base64 string to image bytes
    image_bytes = base64.b64decode(base64_string)
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    # Decode numpy array to OpenCV image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def process_image_from_nodejs(image_base64):
    # Convert base64 image to OpenCV image
    img = process_base64_image(image_base64)
    # Perform predictions
    results = yolo_predictions(img, net)
    # Return results if needed
    return results

import sys

image_base64 = sys.argv[1]

ans = process_image_from_nodejs(image_base64)

print(ans)

sys.stdout.flush()