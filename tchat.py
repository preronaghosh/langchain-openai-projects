# This program is a basic chatbot built on top of OpenAI API. 
# Will retain previous chat history like ChatGPT even after program has been exited.
# Old chat history is saved inside OldMessages.json file.
# Topics: ChatPromptTemplate (Human, System and AI messages), Memory
# To prevent Buffer size explosion, we can use ConersationSummaryMemory instead of ConversationBufferMemory.


from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory

from dotenv import load_dotenv

# to load environment variables
load_dotenv()

chatLLM = ChatOpenAI()
memory = ConversationBufferMemory(
    return_messages=True,  
    memory_key='messages',
    chat_memory=FileChatMessageHistory("OldMessages.json")
)

prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(
            variable_name="messages"
        ),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm=chatLLM,
    prompt=prompt,
    memory=memory
)


# Take user input 
while True:
    content = input(">> ")

    result = chain({"content": content})
    print(result["text"])