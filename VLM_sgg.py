# class for scene graph generation with VLM
import google.generativeai as genai
from ultralytics import YOLO
import supervision as sv
from utils import bboxes_to_rois
import time
import cv2

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

        promt_list = ['You are an object state detector. You are given an image, cropped section of each object in the image, a corresponding list of object names. Based on the given image, list the state of each object in the object list',image]

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

        promt_list = ['You are expected to identify the spatial relationship between objects. You are given an image, cropped section of each object in the image, a list of objects in the image. For each pair of objects in this image, identify the spatial relationship between them.',image]
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

        promt_list = ['You are an object detector. Give a list of all the objects in the image',image]

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
    

    def get_scene_graph(self,image, use_message_history=False):

        message_history_object = []
        message_history_states = []
        message_history_edges = []

        img_1 = cv2.imread('reflect/main/few_shot_data/pic1.png')
        img_2 = cv2.imread('reflect/main/few_shot_data/pic2.png')
        img_3 = cv2.imread('reflect/main/few_shot_data/pic3.png')
        img_4 = cv2.imread('reflect/main/few_shot_data/pic4.png')
        img_5 = cv2.imread('reflect/main/few_shot_data/pic5.png')
        img_6 = cv2.imread('reflect/main/few_shot_data/pic6.png')
        img_7 = cv2.imread('reflect/main/few_shot_data/pic7.png')
        img_8 = cv2.imread('reflect/main/few_shot_data/pic8.png')
        img_9 = cv2.imread('reflect/main/few_shot_data/pic9.png')
        img_10 = cv2.imread('reflect/main/few_shot_data/pic10.png')

        if use_message_history == True:
            message_history_pic1_object = [{'role': 'user', 'parts': ['You are an object detector. Give a list of all the objects in the image',img_1]},
                               {'role': 'model', 'parts': ['[carrot,onion,knife,faucet,stove,mug,countertop,fridge]']}]

            message_history_pic2_object = [{'role': 'user', 'parts': ['You are an object detector. Give a list of all the objects in the image',img_2]},
                               {'role': 'model', 'parts': ['[apple,bowl,cutting board,mug,plate,stove,countertop,fridge]']}]
            message_history_pic3_object  = [{'role': 'user', 'parts': ['You are an object detector. Give a list of all the objects in the image',img_3]},
                               {'role': 'model', 'parts': ['[bowl,steel container,faucet,stove,boxes,mug,countertop,fridge]']}]
            message_history_pic4_object  = [{'role': 'user', 'parts': ['You are an object detector. Give a list of all the objects in the image',img_4]},
                               {'role': 'model', 'parts': ['[tomato,egg,bread,mug,countertop,oven,fridge]']}]
            message_history_pic5_object  = [{'role': 'user', 'parts': ['You are an object detector. Give a list of all the objects in the image',img_5]},
                               {'role': 'model', 'parts': ['[spatula,kettle,salt shaker,pepper shaker,bread loaf,toaster,countertop,plate,microwave]']}]
            message_history_pic6_object  = [{'role': 'user', 'parts': ['You are an object detector. Give a list of all the objects in the image',img_6]},
                               {'role': 'model', 'parts': ['[plant,sofa,table,laptop]']}]
            message_history_pic7_object  = [{'role': 'user', 'parts': ['You are an object detector. Give a list of all the objects in the image',img_7]},
                               {'role': 'model', 'parts': ['[laptop,television,keys,watch,trashcan]']}]
            message_history_pic8_object  = [{'role': 'user', 'parts': ['You are an object detector. Give a list of all the objects in the image',img_8]},
                               {'role': 'model', 'parts': ['[dishwash soap, spatula,knife,plate,faucet,sponge,countertop,bottle]']}]
            message_history_pic9_object  = [{'role': 'user', 'parts': ['You are an object detector. Give a list of all the objects in the image',img_9]},
                               {'role': 'model', 'parts': ['[kettle,egg,pan,salt shaker, pepper shaker]']}]
            message_history_pic10_object  = [{'role': 'user', 'parts': ['You are an object detector. Give a list of all the objects in the image',img_10]},
                               {'role': 'model', 'parts': ['[plants,potato,pan,knife,fork,tomato,bread,lettuce]']}]
            
        
            message_history_object.extend(message_history_pic1_object)
            message_history_object.extend(message_history_pic2_object)
            message_history_object.extend(message_history_pic3_object)
            message_history_object.extend(message_history_pic4_object)
            message_history_object.extend(message_history_pic5_object)
            message_history_object.extend(message_history_pic6_object)
            message_history_object.extend(message_history_pic7_object)
            message_history_object.extend(message_history_pic8_object)
            message_history_object.extend(message_history_pic9_object)
            message_history_object.extend(message_history_pic10_object)
            



            message_history_pic1_states = [{'role': 'user', 'parts': ['You are an object state detector. You are given an image, cropped section of each object in the image, a corresponding list of object names. Based on the given image, list the state of each object in the object list.',img_1]},
                               {'role': 'model', 'parts': ['[faucet:off,stove:off,fridge:closed]']}]

            message_history_pic2_states = [{'role': 'user', 'parts': ['You are an object state detector. You are given an image, cropped section of each object in the image, a corresponding list of object names. Based on the given image, list the state of each object in the object list.',img_2]},
                               {'role': 'model', 'parts': ['[faucet:off,stove:off,fridge:closed']}]
            message_history_pic3_states   = [{'role': 'user', 'parts': ['You are an object state detector. You are given an image, cropped section of each object in the image, a corresponding list of object names. Based on the given image, list the state of each object in the object list.',img_3]},
                               {'role': 'model', 'parts': ['[faucet:off,stove:off]']}]
            message_history_pic4_states  = [{'role': 'user', 'parts': ['You are an object state detector. You are given an image, cropped section of each object in the image, a corresponding list of object names. Based on the given image, list the state of each object in the object list.',img_4]},
                               {'role': 'model', 'parts': ['[oven:off,fridge:closed]']}]
            message_history_pic5_states   = [{'role': 'user', 'parts': ['You are an object state detector. You are given an image, cropped section of each object in the image, a corresponding list of object names. Based on the given image, list the state of each object in the object list.',img_5]},
                               {'role': 'model', 'parts': ['[toaster:on,microwave:on]']}]
            message_history_pic6_states   = [{'role': 'user', 'parts': ['You are an object state detector. You are given an image, cropped section of each object in the image, a corresponding list of object names. Based on the given image, list the state of each object in the object list.',img_6]},
                               {'role': 'model', 'parts': ['[laptop:open]']}]
            message_history_pic7_states  = [{'role': 'user', 'parts': ['You are an object state detector. You are given an image, cropped section of each object in the image, a corresponding list of object names. Based on the given image, list the state of each object in the object list.',img_7]},
                               {'role': 'model', 'parts': ['[laptop:open,television:off]']}]
            message_history_pic8_states  = [{'role': 'user', 'parts': ['You are an object state detector. You are given an image, cropped section of each object in the image, a corresponding list of object names. Based on the given image, list the state of each object in the object list.',img_8]},
                               {'role': 'model', 'parts': ['[faucet:off]']}]
            message_history_pic9_states   = [{'role': 'user', 'parts': ['You are an object state detector. You are given an image, cropped section of each object in the image, a corresponding list of object names. Based on the given image, list the state of each object in the object list.',img_9]},
                               {'role': 'model', 'parts': ['[kettle:off]']}]
            message_history_pic10_states  = [{'role': 'user', 'parts': ['You are an object state detector. You are given an image, cropped section of each object in the image, a corresponding list of object names. Based on the given image, list the state of each object in the object list.',img_10]},
                               {'role': 'model', 'parts': ['[]']}]
          
            message_history_states.extend(message_history_pic1_states)
            message_history_states.extend(message_history_pic2_states)
            message_history_states.extend(message_history_pic3_states)
            message_history_states.extend(message_history_pic4_states)
            message_history_states.extend(message_history_pic5_states)
            message_history_states.extend(message_history_pic6_states)
            message_history_states.extend(message_history_pic7_states)
            message_history_states.extend(message_history_pic8_states)
            message_history_states.extend(message_history_pic9_states)
            message_history_states.extend(message_history_pic10_states)





            message_history_pic1_edges = [{'role': 'user', 'parts': ['You are expected to identify the spatial relationship between objects. You are given an image, cropped section of each object in the image, a list of objects in the image. For each pair of objects in this image, identify the spatial relationship between them.',img_1]},
                               {'role': 'model', 'parts': ['carrot is to the left of the knife.onion is to the right of the carrot.mug is below the countertop.']}]

            message_history_pic2_edges = [{'role': 'user', 'parts': ['You are expected to identify the spatial relationship between objects. You are given an image, cropped section of each object in the image, a list of objects in the image. For each pair of objects in this image, identify the spatial relationship between them.',img_2]},
                               {'role': 'model', 'parts': ['bowl is to the right of the apple.cutting board is hanging on top of the faucet.cutting board is to the right of the apple.plate is above the mug.']}]
            message_history_pic3_edges   = [{'role': 'user', 'parts': ['You are expected to identify the spatial relationship between objects. You are given an image, cropped section of each object in the image, a list of objects in the image. For each pair of objects in this image, identify the spatial relationship between them.',img_3]},
                               {'role': 'model', 'parts': ['bowl is to the left of the steel container.mug is below the countertop.boxes are adjacent to the stove']}]
            message_history_pic4_edges  = [{'role': 'user', 'parts': ['You are expected to identify the spatial relationship between objects. You are given an image, cropped section of each object in the image, a list of objects in the image. For each pair of objects in this image, identify the spatial relationship between them.',img_4]},
                               {'role': 'model', 'parts': ['egg is to the right of tomato.mug is on the countertop.tomato is on the countertop.egg is on the countertop.']}]
            message_history_pic5_edges   = [{'role': 'user', 'parts': ['You are expected to identify the spatial relationship between objects. You are given an image, cropped section of each object in the image, a list of objects in the image. For each pair of objects in this image, identify the spatial relationship between them.',img_5]},
                               {'role': 'model', 'parts': ['bread is inside the toaster.plate is inside the microwave.spatula is on the countertop.kettle is on the countertop.salt shaker is on the countertop.pepper shaker is on the countertop.']}]
            message_history_pic6_edges   = [{'role': 'user', 'parts': ['You are expected to identify the spatial relationship between objects. You are given an image, cropped section of each object in the image, a list of objects in the image. For each pair of objects in this image, identify the spatial relationship between them.',img_6]},
                               {'role': 'model', 'parts': ['plant is on the table.plant is adjacent to the laptop.sofa is in front of laptop.']}]
            message_history_pic7_edges  = [{'role': 'user', 'parts': ['You are expected to identify the spatial relationship between objects. You are given an image, cropped section of each object in the image, a list of objects in the image. For each pair of objects in this image, identify the spatial relationship between them.',img_7]},
                               {'role': 'model', 'parts': ['laptop is in front of television. watch is on  the table. keys are on the table. trashcan is adjacent to the table.']}]
            message_history_pic8_edges  = [{'role': 'user', 'parts': ['You are expected to identify the spatial relationship between objects. You are given an image, cropped section of each object in the image, a list of objects in the image. For each pair of objects in this image, identify the spatial relationship between them.',img_8]},
                               {'role': 'model', 'parts': ['spatula is on the countertop. dishwash soap is on the countertop. knife is in the sink. plate is in the sink. sponge is on the sink. bottle is in the sink.knife is to the right of plate.spatula is in front of dishwashing soap']}]
            message_history_pic9_edges   = [{'role': 'user', 'parts': ['You are expected to identify the spatial relationship between objects. You are given an image, cropped section of each object in the image, a list of objects in the image. For each pair of objects in this image, identify the spatial relationship between them.',img_9]},
                               {'role': 'model', 'parts': ['kettle is on the countertop. egg is on the countertop. pan is on the countertop. salt shaker is on the countertop. pepper shaker is on the countertop. spatula is adjacent to salt shaker. egg is to the left of the pan.']}]
            message_history_pic10_edges  = [{'role': 'user', 'parts': ['You are expected to identify the spatial relationship between objects. You are given an image, cropped section of each object in the image, a list of objects in the image. For each pair of objects in this image, identify the spatial relationship between them.',img_10]},
                               {'role': 'model', 'parts': ['potato is on the pan.lettuce is near the bread.tomato is on the countertop.knife is on the countertop.knife is adjacent to the fork.']}]
            
            
            message_history_edges.extend(message_history_pic1_edges)
            message_history_edges.extend(message_history_pic2_edges)
            message_history_edges.extend(message_history_pic3_edges)
            message_history_edges.extend(message_history_pic4_edges)
            message_history_edges.extend(message_history_pic5_edges)
            message_history_edges.extend(message_history_pic6_edges)
            message_history_edges.extend(message_history_pic7_edges)
            message_history_edges.extend(message_history_pic8_edges)
            message_history_edges.extend(message_history_pic9_edges)
            message_history_edges.extend(message_history_pic10_edges)

            



        objects = self.object_list_detector(image, message_history_object)
        # objects= ['cup','soap_dispenser','potato','sponge','paper_towel','toaster','tap']

        object_bboxes, object_labels = self.object_detector(image, objects)

        rois = bboxes_to_rois(image,object_bboxes)

        states = self.state_detector(image, rois, object_labels, message_history_states)
        time.sleep(1)
        edges = self.edge_detector(image, rois, object_labels, message_history_edges)

        # scene_graph = 'states: '+states+'.'+'edges: '+edges
        scene_graph = states +edges



        return scene_graph


    
