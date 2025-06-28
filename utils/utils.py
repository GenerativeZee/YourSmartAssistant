from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
# from google import genai
import google.generativeai as genai

# from google import genai
import asyncio
import json
import sys
import os
import re

load_dotenv()

api_key = os.getenv("api_key")
model_name = os.getenv("model")

# llm = ChatOpenAI(api_key=api_key, model=model)
import google.generativeai as genai

genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name) 

def callLLM(prompt: str):
   
    response = model.generate_content(prompt)
    
    print("response.text: ", response.text)
    try:
        match = re.search(r"<output>(.*?)</output>", response.text, re.DOTALL | re.IGNORECASE)
        if match:
            output_content = match.group(1).strip()
            print("Extracted content:\n", output_content)
            return output_content
        else:
            return "Unable to process LLM output found"
    except Exception as e:
        return f"Error occurred while fetching the required output from LLM response: {e}"
