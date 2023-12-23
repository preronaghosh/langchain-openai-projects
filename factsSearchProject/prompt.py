# external library imports
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# custom imports
from redundant_filter_retriever import RedundantFilterRetriever

# only for debugging purposes
import langchain
langchain.debug = True

# to load environment variables
load_dotenv()

chatLLM = ChatOpenAI()
embeddings = OpenAIEmbeddings()

# get the existing embeddings from the provided vectorstore
db = Chroma(
    persist_directory='emb',
    embedding_function=embeddings
)

# custom retriever which filters out similar vectors in case of duplicate vectors present 
retriever = RedundantFilterRetriever(
    embeddings=embeddings,
    chroma=db
)

chain = RetrievalQA.from_chain_type(
    llm=chatLLM,
    retriever=retriever,
    chain_type="stuff"
)

result = chain.run("What is an interesting fact about the English language?")
print(result)