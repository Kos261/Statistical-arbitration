# import os
# from dotenv import load_dotenv
# from openai import OpenAI
#
# load_dotenv()
#
# client = OpenAI(
#     # This is the default and can be omitted
#     api_key=os.environ.get("OPENAI_API_KEY"),
# )
#
# response = client.responses.create(
#     model="gpt-5",
#     instructions="You are a coding assistant that talks like a pirate.",
#     input="How do I check if a Python object is an instance of a class?",
# )
#
# print(response.output_text)


import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("API_KEY environment variable is not set")

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-flash')

prompt = "Masz zagadkę: co to jest 'wpadapadanapa'? Zinterpretuj to jako: 'w 'pada' pada na 'pa'"
response = model.generate_content(prompt)

print(response.text)