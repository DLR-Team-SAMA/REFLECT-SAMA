# class for scene graph generation with VLM
import google.generativeai as genai
from ultralytics import YOLO
import supervision as sv
from utils import bboxes_to_rois

class VLM_SGG:
    def __init__(self, model_name):
        self.model = genai.GenerativeModel(model_name)
        self.messages = []
        self.objects = []
        self.rois = []
        self.plan = ''
        self.curr_task = ''
        self.plan_state = False
        self.curr_task_state = False
        self.image = None

    def object_detector(self,image, object_list=None):
        print("Running object detector...")

        model = YOLO('yolov8s-world.pt')
        model.set_classes(object_list)
        results = model.predict(image)
        detections = sv.Detections.from_ultralytics(results[0])
        ret_bboxs = list(detections.xyxy)
        ret_obj_lbls = list(detections.data['class_name'])
        print(ret_bboxs)
        print(ret_obj_lbls)



        return ret_bboxs, ret_obj_lbls
      
    
    def add_message(self, role, parts):
        self.messages.append({'role': role, 'parts': parts})
    
    def state_detector(self, image, rois, objects, message_history=[]):  #Edge - change prompt to edge from states, #Scene graph, object list - remove ROI's and objects
        print('message_history:',message_history)
        model = self.model

        promt_list = ['What is the state are each of these objects in object?',image]

        if self.plan_state:
            promt_list.extend([f"This is the overall plan: {self.plan}", self.plan])

        if self.curr_task_state:
            promt_list.extend([f"This is the current task: {self.curr_task}", self.curr_task])
        if(len(rois) != len(objects)):
            print('Error: Number of objects and ROIs do not match')
            return
        for i in range(len(rois)):
            promt_list.append(objects[i])
            promt_list.append(rois[i])
        messages = message_history
        print("messages:",messages) 
        messages.append({'role': 'user', 'parts':promt_list})
        response = model.generate_content(messages[0]['parts'])
        return response.text

    def edge_detector(self, image, rois, objects, message_history=[]):  
        print('message_history:',message_history)
        model = self.model

        promt_list = ['What are the relationships/edges between the objects?',image]
        if self.plan_state:
            promt_list.extend([f"This is the overall plan: {self.plan}", self.plan])

        if self.curr_task_state:
            promt_list.extend([f"This is the current task: {self.curr_task}", self.curr_task])
            
        if(len(rois) != len(objects)):
            print('Error: Number of objects and ROIs do not match')
            return
        for i in range(len(rois)):
            promt_list.append(objects[i])
            promt_list.append(rois[i])
        messages = message_history
        print("messages:",messages) 
        messages.append({'role': 'user', 'parts':promt_list})
        response = model.generate_content(messages[0]['parts'])
        return response.text

    def object_list_detector(self, image, message_history=[]): 
        print('message_history:',message_history)
        model = self.model

        promt_list = ['Give a list of objects present in the scene?',image]

        if self.plan_state:
            promt_list.extend([f"This is the overall plan: {self.plan}", self.plan])

        if self.curr_task_state:
            promt_list.extend([f"This is the current task: {self.curr_task}", self.curr_task])

        messages = message_history
        print("messages:",messages) 
        messages.append({'role': 'user', 'parts':promt_list})
        response = model.generate_content(messages[0]['parts'])
        object_list = response.text.strip().split('\n')  # Splitting based on new lines to handle each item
        cleaned_object_list = [obj.strip('* ').replace('A ', '').replace('An ', '') for obj in object_list if obj.strip() ]

        cleaned_object_list = [obj for obj in cleaned_object_list if obj] 
        return cleaned_object_list
    
    def e2e_sgg(self,image, message_history=[]):  
        print('message_history:',message_history)
        model = self.model
        promt_list = ['Generate a scene graph given this image?',image]

        if self.plan_state:
            promt_list.extend([f"This is the overall plan: {self.plan}", self.plan])

        if self.curr_task_state:
            promt_list.extend([f"This is the current task: {self.curr_task}", self.curr_task])
        messages = message_history
        print("messages:",messages) 
        messages.append({'role': 'user', 'parts':promt_list})
        response = model.generate_content(messages[0]['parts'])

        return response.text
    
    def sgg_layered(self, objects, states, edges, message_history=[]): 
        print('message_history:',message_history)
        model = self.model

        promt_list = ['Construct a scene graph with the given objects, states and edges. \n Objects:'+objects+'\n States:'+states+'\n Edges:'+edges]

        if self.plan_state:
            promt_list.extend([f"This is the overall plan: {self.plan}", self.plan])

        if self.curr_task_state:
            promt_list.extend([f"This is the current task: {self.curr_task}", self.curr_task])

        messages = message_history
        print("messages:",messages) 
        messages.append({'role': 'user', 'parts':promt_list})
        response = model.generate_content(messages[0]['parts'])
        return response.text
    def get_scene_graph(self,image):
        objects = self.object_list_detector(image)
        # objects= ['cup','soap_dispenser','potato','sponge','paper_towel','toaster','tap']

        object_bboxes, object_labels = self.object_detector(image, objects)

        rois = bboxes_to_rois(image,object_bboxes)

        states = self.state_detector(image, rois, object_labels, [])
        edges = self.edge_detector(image, rois, object_labels, [])

        scene_graph = 'states: '+states+'\n'+'edges: '+edges

        return scene_graph


    
