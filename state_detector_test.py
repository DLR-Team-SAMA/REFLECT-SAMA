import pathlib
import textwrap

import google.generativeai as genai

import os

import PIL.Image
import PIL.ImageDraw

def state_detector(vlm_model, image,roi,plan,curr_task):
  model = vlm_model
  promt_list = ['What is the state of this object?',image,roi]
  response = model.generate_content(promt_list)
  print(response.text)

if __name__ == '__main__':
  GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')

  genai.configure(api_key=GOOGLE_API_KEY)
  for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
      print(m.name)

  # model = genai.GenerativeModel('models/gemini-pro-vision')
  model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
  img_fn = 'test_images/frame_52.jpg'
  image = PIL.Image.open(img_fn)
  messages = [{'role': 'user', 'parts':['Generate a scene graph for this image?', image]}]
  
  # messages = [{'role': 'user', 'parts':['what are the objects of interest in the scene?', image]}]
  # promt_list = ['what are the obejcts of interest in the scene?',image]
  # response = model.generate_content(promt_list)
  # response = model.generate_content(messages)
  # print(response.text)
  # message = {'role':'model','parts':[response.text]}
  # messages.append(message)
  # message = {'role':'user','parts':['What a the spatial relationships between each pair of objects?',image]}
  # messages.append(message)
  response = model.generate_content(messages)
  print(response.text)

  bbox = [430,440, 515, 550]
  # bbox = [605,400,725,520]
  # draw the bounding box
  roi = image.crop(bbox)

  # draw = PIL.ImageDraw.Draw(image)
  # draw.rectangle(bbox, outline='red', width=2)
  # image.show()


  # roi.save('roi.jpg')
