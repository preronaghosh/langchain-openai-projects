# This code will read facts from facts.txt
# Based on user's question, it will search the most relevant fact
# Create a prompt with that information
# Return a response form LLM based on the most relevant fact
# Topics: Document loaders, Searching, Embeddings

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv

# to load environment variables
load_dotenv()

# part of data preprocessing
# create embeddings of a sentence/text
embeddings = OpenAIEmbeddings()

# first apply the chunk size and then separator
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0
)

loader = TextLoader("facts.txt")
docs = loader.load_and_split(
    text_splitter=text_splitter
)

# Generate embeddings and store it into vector database
# When run more than once, duplicate records will be created inside the vector store
db = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory="emb"
)

# Sample test run for debugging after creating vector store initially
# results = db.similarity_search(
#     "What is the most interesting fact about English language?"
# )

# for res in results:
#     print("\n")
#     print(res.page_content)
