class DummySGG:
    def __init__(self) -> None:
        print("Dummy SGG object created")
    
    def object_detector(self,image, object_list=None, img_fn = 'test_images/frame_15.jpg'):
        print("Running dummy object detector...")
        if(img_fn == 'test_images/frame_15.jpg'):
            if(object_list is None):
                object_list= ['cup','soap_dispenser','potato','sponge','paper_towel','toaster','tap']
            bbox_cup = [430,440, 515, 550]
            bbox_soap_disp = [604, 207, 725, 416]
            bbox_potato = [749,425,893,548]
            bbox_sponge = [44, 270, 167, 320]
            bbox_ppr_twl = [590,119,658,290]
            bbox_toaster = [791,97,961,380]
            bbox_tap = [210,199,300,334]
            
            bboxes = [bbox_cup,bbox_soap_disp,bbox_potato,bbox_sponge,bbox_ppr_twl,bbox_toaster,bbox_tap]
            objects_lables = ['cup','soap_dispenser','potato','sponge','paper_towel','toaster','tap']
            ret_bboxs = []
            ret_obj_lbls = []
            for i in range(len(bboxes)):
                if objects_lables[i] in object_list:
                    ret_bboxs.append(bboxes[i])
                    ret_obj_lbls.append(objects_lables[i])
            return ret_bboxs, ret_obj_lbls
        raise Exception('Image not handled by dummy object detector')
        
    def object_list_detector(self,image, img_fn = 'test_images/frame_15.jpg'):
        print("Running dummy object detector...")
        if(img_fn == 'test_images/frame_15.jpg'):
            object_list= ['cup','soap_dispenser','potato','sponge','paper_towel','toaster','tap']
            return object_list
        raise Exception('Image not handled by dummy object list detector')
        
    def edge_detector(self, image, rois, message_history=[], img_fn = 'test_images/frame_15.jpg'):
        print("Running dummy edge detector...")
        if(img_fn == 'test_images/frame_15.jpg'):
            edges = '''cup is in the sink
soap_dispenser is on the counter
potato is on the counter
sponge is on the counter
paper_towel is on the counter
toaster is on the counter'''
            return edges
        raise Exception('Image not handled by dummy edge detector')

    def state_detector(self, image, rois, message_history=[], img_fn = 'test_images/frame_15.jpg'):
        print("Running dummy state detector...")
        if(img_fn == 'test_images/frame_15.jpg'):
            states = '''cup is empty
soap_dispenser is full
potato is unpeeled
sponge seems dry
paper_towel is unused
toaster is off'''
            return states
        raise Exception('Image not handled by dummy state detector')
    
    def sgg_layered(self, objects, states, edges, message_history=[],img_fn = 'test_images/frame_15.jpg'):
        print("Running dummy sgg layered...")
        if(img_fn == 'test_images/frame_15.jpg'):
            sgg = "The objects in the scene are:"+objects+"\n"+"The object states are:"+states+"\nThe realtion between the objects are:"+edges
            return sgg
        
        raise Exception('Image not handled by dummy sgg layered')

