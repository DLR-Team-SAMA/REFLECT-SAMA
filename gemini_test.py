import google.generativeai as genai

import os


GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')
print(GOOGLE_API_KEY)
genai.configure(api_key=GOOGLE_API_KEY)
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)
