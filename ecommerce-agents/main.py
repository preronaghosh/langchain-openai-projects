# external library imports
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# custom imports
from tools.sql import run_query_tool, list_tables, describe_tables_tool
from tools.report import write_report_tool
from handlers.chat_model_start_handler import ChatModelStartHandler

load_dotenv()

handler = ChatModelStartHandler()
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
tools = [run_query_tool, describe_tables_tool, write_report_tool]
chatLLM = ChatOpenAI(
    callbacks=[handler]
)

tables = list_tables()
prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(content=(
            "You are an AI that has access to a SQLITE database.\n"
            f"The database only has the following tables: {tables}\n"
            "Do not make any assumptions about what tables or columns exist. Instead, use the 'describe_tables' function."
        )),
        MessagesPlaceholder(variable_name="chat_history"), # to preserve memory across agent executor queries
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
) 

agent = OpenAIFunctionsAgent(
    llm=chatLLM,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(
    agent=agent,
    # verbose=True,
    tools=tools,
    memory=memory
)
agent_executor("Summarize the top 5 products. Write the summary to a report.")