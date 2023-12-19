# Program 1
# Generates a code and a unit test for it
# Topics: PromptTemplate, Chains, basic API usage

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

from dotenv import load_dotenv
import argparse

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument('--task', default="generate a code to check even and odd numbers")
parser.add_argument('--language', default='c++')
args = parser.parse_args()

llm = OpenAI()

code_prompt = PromptTemplate(
    input_variables=['language', 'task'],
    template="Write a {language} code for geneating {task}"
)

code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key="code"
)

test_prompt = PromptTemplate(
    input_variables=["language", "code"],
    template="Write a {language} unit test for this code: {code}"
)

test_chain = LLMChain(
    llm=llm,
    prompt=test_prompt,
    output_key="test"
)

chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["task", "language"],
    output_variables=["code", "test"]
)

result = chain({
    "task": args.task,
    "language": args.language
})


print(">>>>> GENERATED CODE: ")
print(result["code"])

print(">>>>> GENERATED TEST: ")
print(result["test"])