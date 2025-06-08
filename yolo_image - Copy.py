import cv2
import numpy as np

# File paths (just change image file name here if needed)
image_path = r"E:\ML\object-detection-opencv-master\kolo1.jpg"
config_path = r"E:\ML\object-detection-opencv-master\yolov3.cfg"
weights_path = r"E:\ML\object-detection-opencv-master\yolov3.weights"
classes_path = r"E:\ML\object-detection-opencv-master\yolov3.txt"  # can be coco.names

# Load YOLO
net = cv2.dnn.readNet(weights_path, config_path)

# Load image
image = cv2.imread(image_path)
Height, Width = image.shape[:2]
scale = 0.00392

# Load classes
with open(classes_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Generate random colors
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Get output layer names
def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        return [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        return [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw bounding box
def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Prepare input blob
blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
net.setInput(blob)

# Forward pass
outs = net.forward(get_output_layers(net))

# Post-processing
class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

for i in indices:
    i = i[0] if isinstance(i, (tuple, list, np.ndarray)) else i
    box = boxes[i]
    x, y, w, h = box
    draw_prediction(image, class_ids[i], confidences[i], x, y, x + w, y + h)

# Show image
cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save result
cv2.imwrite("det.jpg", image)
