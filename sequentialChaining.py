from perplexity_ai_llm import PerplexityAILLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
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

test_prompt = PromptTemplate(
  template="Write a test for the following {language} :\n {code}",
  input_variables=["code","language"]
)

# making a chain

code_chain = LLMChain(
  llm = llm,
  prompt=code_prompt,
  output_key="code"
)

test_chain = LLMChain(
  llm = llm,
  prompt=test_prompt,
  output_key="test"
)

chain = SequentialChain(
  chains = [code_chain,test_chain],
  input_variables=["task","language"],
  output_variables=["test","code"]
)

# inputting a dictionary
result = chain(
  {
    "language":args.language,
    "task":args.task
  }
)

print(">>>> GENERATED CODE: ")
print(result["code"])
print(">>>> GENERATED TEST: ")
print(result["test"])
# this is going to be a dictionary

# a simple chain in LangChain or LLMChain consists of a LLM + A Prompt Template


# creating a second prompt template

