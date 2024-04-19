from dummy import DummySGG
from utils import bboxes_to_rois
import PIL.Image

sgg1 = DummySGG()
sgg2 = sgg1

img = PIL.Image.open('test_images/frame_15.jpg')
# get object list
object_list = sgg1.object_list_detector('test_images/frame_15.jpg')
print(object_list)

# get object bounding boxes
object_bboxes, object_labels = sgg2.object_detector(img, object_list,'test_images/frame_15.jpg')
print(object_labels)
print(object_bboxes)

rois = bboxes_to_rois(img,object_bboxes)

# get states 
states = sgg1.state_detector(img, rois, [], 'test_images/frame_15.jpg')
print(states)

# get edges
edges = sgg1.edge_detector(img, rois, [], 'test_images/frame_15.jpg')
print(edges)

# get scene graph