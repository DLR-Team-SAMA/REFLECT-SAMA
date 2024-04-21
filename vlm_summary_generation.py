from VLM_sgg import VLM_SGG
import cv2
from PIL import Image
import os
import time

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

    summary_L2_path = f'{data_fold}/vlm_summary_L2.txt'
    

    common_frames = set(event_keyframes).intersection(frame_numbers)
    with open(summary_L2_path, 'w') as file:
        for i, frame in enumerate(frame_numbers):
            if frame in common_frames:
                file.write(captions1[i] + '\n')  

    print(f"Output written to {summary_L2_path}")



        








def get_vlm_summary(vlm_sgg, data_fold):

    summary_file = f'{data_fold}/vlm_summary_L1.txt'

    # Check if the summary file already exists
    # if os.path.exists(summary_file):
    #     print("Summary file already exists. Skipping processing.")
    #     return


    vid_path = f'{data_fold}/original-video.mp4'
    kf_event = f'{data_fold}/keyframes_event.txt'
    kf_subgoal = f'{data_fold}/keyframes_subgoal.txt'
    task_json = f'{data_fold}/task.json'
    cap = cv2.VideoCapture(vid_path)
    event_keyframes = []
    subgoal_keyframes = []
    with open(kf_event,'r') as f:
        captions = f.read().strip().split('\n')
        event_keyframes = [int(kf.split(',')[0]) for kf in captions]
    print(event_keyframes)
    out = ''
    for kf,cptn in zip(event_keyframes, captions):
        print(cptn,"cptn")
        cap.set(cv2.CAP_PROP_POS_FRAMES, kf)
        ret, frame = cap.read()
        color_coverted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(color_coverted)
        vis_obs = vlm_sgg.get_scene_graph(frame_pil)
        vis_obs = vis_obs.replace('\n','')
        tmp_cap = cptn.split(',')
        print(tmp_cap,"tmp_cap")
        tmp_cap[0] = frame_to_sec(tmp_cap[0])
        tmp_caps = tmp_cap[0]+'.'+tmp_cap[1]
        out+=f'{tmp_caps}'+f'. Visual observation:{vis_obs}'+'\n'
        print(f'{tmp_caps}'+f'Visual observation:{vis_obs}'+'\n')
        print("Vis_obs:")
        print(vis_obs)
        cv2.imshow('frame',frame)
        cv2.waitKey(60000)
        # time.sleep(60)
    f = open(f'{data_fold}/vlm_summary_L1.txt','w')
    f.write(out)
    f.close()

# fld = 'reflect/main/keyframe_dataset/boilWater/boilWater-1'
# get_vlm_summary(sgg,fld)

# kframe_dataset = 'reflect/main/keyframe_dataset'
# tasks = os.listdir(kframe_dataset)
# for task in tasks:
#     cases = os.listdir(f'{kframe_dataset}/{task}')
#     for case in cases:
#         print(f'{kframe_dataset}/{task}/{case}')
#         get_vlm_summary(sgg,f'{kframe_dataset}/{task}/{case}')


# kframe_dataset = 'reflect/main/keyframe_dataset'
# tasks = os.listdir(kframe_dataset)
# for task in tasks:
#     cases = os.listdir(f'{kframe_dataset}/{task}')
#     for case in cases:
#         print(f'{kframe_dataset}/{task}/{case}')
#         get_L2_summary(f'{kframe_dataset}/{task}/{case}')


get_vlm_summary(sgg,'reflect/main/keyframe_dataset/boilWater/boilWater-1')
get_L2_summary('reflect/main/keyframe_dataset/boilWater/boilWater-1')
    
