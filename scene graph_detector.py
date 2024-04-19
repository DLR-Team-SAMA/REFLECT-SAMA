import pathlib
import textwrap

import google.generativeai as genai

import os

import PIL.Image
import PIL.ImageDraw

def e2e_sgg(vlm_model, image, plan, plan_state, curr_task, curr_task_state, message_history=[]):  
  print('message_history:',message_history)
  model = vlm_model
  promt_list = ['Generate a scene graph given this image?',image]

  if plan_state:
    promt_list.extend([f"This is the overall plan: {plan}", plan])

  if curr_task_state:
    promt_list.extend([f"This is the current task: {curr_task}", curr_task])
  messages = message_history
  print("messages:",messages) 
  messages.append({'role': 'user', 'parts':promt_list})
  response = model.generate_content(messages[0]['parts'])
  return response.text

if __name__ == '__main__':
  GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')

  genai.configure(api_key=GOOGLE_API_KEY)
  for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
      print(m.name)

  # model = genai.GenerativeModel('models/gemini-pro-vision')
  model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
  img_fn = 'test_images/frame_15.jpg'
  image = PIL.Image.open(img_fn)
  # messages = [{'role': 'user', 'parts':['Generate a scene graph for this image?', image]}]
  
  # messages = [{'role': 'user', 'parts':['what are the objects of interest in the scene?', image]}]
  # promt_list = ['what are the obejcts of interest in the scene?',image]
  # response = model.generate_content(promt_list)
  # response = model.generate_content(messages)
  # print(response.text)
  # message = {'role':'model','parts':[response.text]}
  # messages.append(message)
  # message = {'role':'user','parts':['What a the spatial relationships between each pair of objects?',image]}
  # messages.append(message)
  # response = model.generate_content(messages)
  # print(response.text)

  bbox_cup = [430,440, 515, 550]
  bbox_soap_disp = [604, 207, 725, 416]
  bbox_potato = [749,425,893,548]
  bbox_sponge = [44, 270, 167, 320]
  bbox_ppr_twl = [590,119,658,290]
  bbox_toaster = [791,97,961,380]
  bbox_tap = [210,199,300,334]


  # bbox = [605,400,725,520]
  # draw the bounding box

  # draw = PIL.ImageDraw.Draw(image)
  # draw.rectangle(bbox, outline='red', width=2)
  # image.show()

  plan_state = False
  curr_task_state = False
  rsp = e2e_sgg(model, image,  'plan',plan_state, 'curr_task', curr_task_state)

  print(rsp)
  # roi.save('roi.jpg')
