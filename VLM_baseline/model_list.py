import pathlib
import textwrap

import google.generativeai as genai

import os

import PIL.Image



GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')
GOOGLE_API_KEY = 'AIzaSyAJZJfvHaoZ7peAEOcr2g-rmXrJF0rdxww'

genai.configure(api_key=GOOGLE_API_KEY)
for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)