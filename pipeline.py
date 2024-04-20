from VLM_sgg import VLM_SGG
from utils import bboxes_to_rois
import PIL.Image
from dummy import DummySGG

sgg2 = DummySGG()
sgg1 = VLM_SGG('models/gemini-1.5-pro-latest')

img = PIL.Image.open('test_images/frame_15.jpg')
# get object list
object_list = sgg1.object_list_detector(img)
print(object_list)

# get object bounding boxes
object_bboxes, object_labels = sgg2.object_detector(img, object_list)
print(object_labels)
print(object_bboxes)

rois = bboxes_to_rois(img,object_bboxes)

# get states 
states = sgg1.state_detector(img, rois, object_labels, [])
print(states)

# get edges
edges = sgg1.edge_detector(img, rois, object_labels, [])
print(edges)

# get scene graph
