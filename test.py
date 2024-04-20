import PIL.Image
from VLM_sgg import VLM_SGG

img = PIL.Image.open('test_images/frame_15.jpg')

sgg = VLM_SGG('models/gemini-1.5-pro-latest')

# get scene graph
scene_graph = sgg.get_scene_graph(img)
print(scene_graph)