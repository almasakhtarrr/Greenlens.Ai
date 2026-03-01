import cv2
import numpy as np

def run_inference(image, model_path="models/yolov8n.onnx"):
    # Load ONNX model via OpenCV DNN
    net = cv2.dnn.readNetFromONNX(model_path)
    
    # YOLOv8 input size is 640x640
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Forward pass: shape [1, 84, 8400]
    outputs = net.forward()
    
    rows = outputs[0].T
    boxes, confidences, class_ids = [], [], []
    
    # Image dimensions for scaling boxes
    img_h, img_w = image.shape[:2]
    x_factor = img_w / 640
    y_factor = img_h / 640

    for row in rows:
        confidence = row[4:].max()
        if confidence > 0.4:
            class_id = row[4:].argmax()
            cx, cy, w, h = row[0:4]
            # Rescale to original image size
            left = int((cx - 0.5 * w) * x_factor)
            top = int((cy - 0.5 * h) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            
            boxes.append([left, top, width, height])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    # Remove overlapping boxes (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.4)
    
    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            detections.append({
                "box": boxes[i],
                "conf": confidences[i],
                "class_id": class_ids[i]
            })
    return detections

def draw_boxes(image, detections, classes):
    for det in detections:
        x, y, w, h = det['box']
        label = f"{classes[det['class_id']]} {det['conf']:.2f}"
        # Green box for eco-friendly feel
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return image
