import urllib.request

# New GitHub mirror links
weights_url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov3.weights"
cfg_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3.cfg"
names_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

print("🔽 Downloading yolov3.weights...")
urllib.request.urlretrieve(weights_url, "yolov3.weights")

print("🔽 Downloading yolov3.cfg...")
urllib.request.urlretrieve(cfg_url, "yolov3.cfg")

print("🔽 Downloading coco.names as yolov3.txt...")
urllib.request.urlretrieve(names_url, "yolov3.txt")

print("✅ All files downloaded successfully.")
