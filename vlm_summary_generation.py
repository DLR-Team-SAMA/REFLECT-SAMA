from VLM_sgg import VLM_SGG
import cv2
from PIL import Image
import os
import time

class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                f.write('')
            self.last_frame = -1
        else:
            self.last_frame = self.get_last_frame()
    def add_log(self, log):
        with open(self.log_file, 'a') as f:
            f.write(log)
    def get_text(self): 
        with open(self.log_file, 'r') as f:
            return f.read()
    def get_last_frame(self):
        if not os.path.exists(self.log_file):
            return -1
        with open(self.log_file, 'r') as f:
            lines = f.readlines()
            if len(lines) > 0:
                if(lines[-1] == '\n'):
                    lines = lines[:-1]
            if len(lines) > 0:
                last_line = lines[-1]
                time_stmp = last_line.split('.')[0]
                tm_cmps = time_stmp.split(':')
                return int(tm_cmps[0])*60 + int(tm_cmps[1])
            else:
                return -1
    
    



def frame_to_sec(frame):
    frame = int(frame)
    minutes = str(frame//60)
    minutes.zfill(2)
    seconds = str(frame%60)
    seconds.zfill(2)
    return f'{minutes}:{seconds}'

def sec_to_frame(sec):
    sec = sec.split(':')
    return int(sec[0])*60 + int(sec[1])


sgg = VLM_SGG('models/gemini-1.5-pro-latest')

def get_L2_summary(data_fold):
    summary_L2_path = f'{data_fold}/vlm_summary_L2.txt'
    # Check if the summary file already exists
    if os.path.exists(summary_L2_path):
        print("Summary file already exists. Skipping processing.")
        return

    kf_subgoal = f'{data_fold}/keyframes_subgoal.txt'
    vlm_summ = f'{data_fold}/vlm_summary_L1.txt'
    event_keyframes = []
    event_keyframes1 = []
    with open(kf_subgoal,'r') as f:
        captions = f.read().strip().split('\n')
        event_keyframes = [int(kf.split(',')[0]) for kf in captions]
        # summary = [kf.split(',')[1] for kf in captions]
    print(event_keyframes)
    
 
    with open(vlm_summ,'r') as f:
        captions1 = f.read().strip().split('\n')
        event_keyframes1 = [kf.split('.')[0]for kf in captions1]
        summary1 = [kf.split('.')[1] for kf in captions1]
    print(event_keyframes1,"event_keyframes1")
    print(summary1,"summary1")

    frame_numbers = [sec_to_frame(time) for time in event_keyframes1]

    common_frames = set(event_keyframes).intersection(frame_numbers)
    with open(summary_L2_path, 'w') as file:
        for i, frame in enumerate(frame_numbers):
            if frame in common_frames:
                file.write(captions1[i] + '\n')  

    print(f"Output written to {summary_L2_path}")



def get_vlm_summary(vlm_sgg, data_fold):

    summary_file = f'{data_fold}/vlm_summary_L1.txt'

    # Check if the summary file already exists
    if os.path.exists(summary_file):
        print("Summary file already exists. Skipping processing.")
        return
    else:
        print("Processing new summary for:", data_fold)

    log_file = f'{data_fold}/log.txt'
    vid_path = f'{data_fold}/original-video.mp4'
    kf_event = f'{data_fold}/keyframes_event.txt'
    kf_subgoal = f'{data_fold}/keyframes_subgoal.txt'
    task_json = f'{data_fold}/task.json'
    cap = cv2.VideoCapture(vid_path)
    event_keyframes = []
    subgoal_keyframes = []
    
    logger = Logger(log_file)


    with open(kf_event,'r') as f:
        captions = f.read().strip().split('\n')
        event_keyframes = [int(kf.split(',')[0]) for kf in captions]
    # print(event_keyframes)
    # if(logger.last_frame != -1):
    #     event_keyframes = [frame for frame in event_keyframes if frame > logger.last_frame]
    print(event_keyframes)

    out = logger.get_text()

    for kf,cptn in zip(event_keyframes, captions):
        if kf <= logger.last_frame:
            print(f"Skipping frame {kf} as it has already been processed.")
            continue

        print(cptn,"cptn")
        cap.set(cv2.CAP_PROP_POS_FRAMES, kf)
        ret, frame = cap.read()
        color_coverted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(color_coverted)
        # while True:
        #     try:
        #         vis_obs = vlm_sgg.get_scene_graph(frame_pil)
        #         break
        #     except Exception as e:
        #         print(e)
        #         print("Retrying")
        #         time.sleep(100)
        #         continue
        vis_obs = vlm_sgg.get_scene_graph(frame_pil)
        vis_obs = vis_obs.replace('\n','')
        tmp_cap = cptn.split(',')
        print(tmp_cap,"tmp_cap")
        tmp_cap[0] = frame_to_sec(tmp_cap[0])
        tmp_caps = tmp_cap[0]+'.'+tmp_cap[1]
        out+=f'{tmp_caps}'+f'. Visual observation:{vis_obs}'+'\n'
        logger.add_log(f'{tmp_caps}'+f'. Visual observation:{vis_obs}'+'\n')
        print(f'{tmp_caps}'+f'Visual observation:{vis_obs}'+'\n')
        print("Vis_obs:")
        print(vis_obs)
        cv2.imshow('frame',frame)
        cv2.waitKey(1)
        # time.sleep(60)
    f = open(f'{data_fold}/vlm_summary_L1.txt','w')
    f.write(out)
    f.close()

# fld = 'reflect/main/keyframe_dataset/boilWater/boilWater-1'
# get_vlm_summary(sgg,fld)

kframe_dataset = 'reflect/main/keyframe_dataset'
tasks = os.listdir(kframe_dataset)
for task in tasks:
    cases = os.listdir(f'{kframe_dataset}/{task}')
    for case in cases:
        print(f'{kframe_dataset}/{task}/{case}')
        get_vlm_summary(sgg,f'{kframe_dataset}/{task}/{case}')


kframe_dataset = 'reflect/main/keyframe_dataset'
tasks = os.listdir(kframe_dataset)
for task in tasks:
    cases = os.listdir(f'{kframe_dataset}/{task}')
    for case in cases:
        print(f'{kframe_dataset}/{task}/{case}')
        get_L2_summary(f'{kframe_dataset}/{task}/{case}')


# get_vlm_summary(sgg,'reflect/main/keyframe_dataset/boilWater/boilWater-1')
# get_L2_summary('reflect/main/keyframe_dataset/boilWater/boilWater-1')
    
