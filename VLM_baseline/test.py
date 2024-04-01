import pathlib
import textwrap

import google.generativeai as genai

import os

import PIL.Image



GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')

genai.configure(api_key=GOOGLE_API_KEY)
for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)

model = genai.GenerativeModel('gemini-1.0-pro-vision-latest')

img = PIL.Image.open('data/images/frame_0.jpg')
img2 = PIL.Image.open('data/images/frame_3.jpg')

# importing images
imgs=[]
for i in range(0, 32,2):
  imgs.append(PIL.Image.open(f'data/images/frame_{i}.jpg'))
  print(i)

text = '''A robot is executing the following task plan:
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
            
            Which subtask have been completed?'''

promt_list = [text]
promt_list.extend(imgs)
response = model.generate_content(promt_list)
print(response.text)

# display the last image

img = imgs[-1]
img.show()
