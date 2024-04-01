import pickle as pkl
import os
import PIL.Image

import pathlib
import textwrap

import google.generativeai as genai

def sort_sub_task_flds(sub_task_flds):
    sub_task_flds.sort(key=lambda x: int(x.split('_')[0]))
    return sub_task_flds

images = os.listdir('data/images')

sub_task_flds = os.listdir('data/images')
sub_task_flds = sort_sub_task_flds(sub_task_flds)
print(sub_task_flds)

# print(sub_task_flds)

def sort_frame_list(frame_list):
    frame_list.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    return frame_list
text = '''A task plan is being executed in AI2Thor simulation environment:
            1 (navigate_to_obj, Mug),
            2 (pick_up, Mug),
            3 (navigate_to_obj, Sink),
            4 (put_on, Mug, SinkBasin),
            5 (toggle_on, Faucet),
            6 (toggle_off, Faucet),
            7 (pick_up, Mug),
            8 (pour, Mug, Sink),
            9 (navigate_to_obj, CoffeeMachine),
            10 (put_in, Mug, CoffeeMachine),
            11 (toggle_on, CoffeeMachine),
            12 (toggle_off, CoffeeMachine),
            13 (pick_up, Mug),
            14 (put_on, Mug, CounterTop)
            
            This is the scene after executing the subtask: '''

questions = '\nHas the subtask been completed successfully? <yes> or <no>. If no please provide a reason. \n'



sub_tasks = [
        "(navigate_to_obj, Mug)",
        "(pick_up, Mug)",
        "(navigate_to_obj, Sink)",
        "(put_on, Mug, SinkBasin)",
        "(toggle_on, Faucet)",
        "(toggle_off, Faucet)",
        "(pick_up, Mug)",
        "(pour, Mug, Sink)",
        "(navigate_to_obj, CoffeeMachine)",
        "(put_in, Mug, CoffeeMachine)",
        "(toggle_on, CoffeeMachine)",
        "(toggle_off, CoffeeMachine)",
        "(pick_up, Mug)",
        "(put_on, Mug, CounterTop)"]

GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-1.0-pro-vision-latest')
for tsk,tsk_fld in zip(sub_tasks,sub_task_flds):
    print(f'task_fld: {tsk_fld} , task: {tsk}')
    img_fls = os.listdir('data/images/'+tsk_fld)
    img_fls=sort_frame_list(img_fls)
    print(img_fls)
    imgs = []

    for img_fl in img_fls:
        imgs.append(PIL.Image.open('data/images/'+tsk_fld+'/'+img_fl))
    print(tsk)
    promt_list = [text+tsk+questions]
    # promt_list.extend(imgs)
    promt_list.append(imgs[-1])
    response = model.generate_content(promt_list)
    print(response.text)
    print("#"*50)
    