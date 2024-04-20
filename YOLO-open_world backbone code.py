from ultralytics import YOLO
import supervision as sv
# import cv2 

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
img = 'test_images/frame_52.jpg'
model = YOLO('yolov8s-world.pt')
model.set_classes(['mug','coffee_machine'])
results = model.predict(img)

results[0].show()

detections = sv.Detections.from_ultralytics(results[0])
# annotated_frame = bounding_box_annotator.annotate(scene = img.copy(), detections=detections)
# annotated_frame = label_annotator.annotate(scene = annotated_frame, detections = detections)

