import cv2
import numpy as np

# Load YOLO
def load_yolo():
    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
    layer_names = net.getLayerNames()
    
    try:
        # Try accessing as if the output is a list of tuples (common in older versions)
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    except Exception:
        # If it fails, treat it as a flat array (common in newer versions)
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    return net, output_layers

# Function to find pixel differences between two images
def find_pixel_difference(img1, img2):
    diff = cv2.absdiff(img1, img2)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
    return threshold

# Detect objects using YOLO
def detect_objects(img, net, output_layers):
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    return outputs

# Get bounding boxes from the YOLO output
def get_boxes(outputs, width, height):
    boxes = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            #print(scores)
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:  # Lowered confidence for testing
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
    return boxes

# Draw bounding boxes on the image
def draw_boxes(img, boxes):
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Main processing of the video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    prev_img = None
    net, output_layers = load_yolo()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if prev_img is not None:
            thres = find_pixel_difference(prev_img, frame)
            masked_frame = cv2.bitwise_and(frame, frame, mask=thres)
            
            outputs = detect_objects(masked_frame, net, output_layers)
            boxes = get_boxes(outputs, frame.shape[1], frame.shape[0])
            print(boxes)
            draw_boxes(frame, boxes)

            cv2.imshow('Frame with Detected Changes', frame)

            if cv2.waitKey(10000) & 0xFF == ord('q'):
                break

        prev_img = frame

    cap.release()
    cv2.destroyAllWindows()

# Replace 'input_video.mp4' with your video file path
process_video('color1.mp4')