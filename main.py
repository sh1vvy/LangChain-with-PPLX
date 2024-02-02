from perplexity_ai_llm import PerplexityAILLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from config import MY_API_KEY

from dotenv import load_dotenv

import argparse

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--task",default="return a list of numbers")
parser.add_argument("--language",default="python")

args = parser.parse_args()

llm = PerplexityAILLM(api_key=MY_API_KEY,model_name="pplx-70b-online")

code_prompt = PromptTemplate(
  template = "Write a very short {language} function that will {task}",
  input_variables=["language","task"]
)

# making a chain

code_chain = LLMChain(
  llm = llm,
  prompt=code_prompt
)

# inputting a dictionary
result = code_chain(
  {
    "language":args.language,
    "task":args.task
  }
)

print(result["text"])

# this is going to be a dictionary

# a simple chain in LangChain or LLMChain consists of a LLM + A Prompt Template
