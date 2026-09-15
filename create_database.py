from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
load_dotenv()

from langchain_core.documents import Document

data = PyPDFLoader("documents/deeplearningbyOrelly.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=10
)

chunks = splitter.split_documents(docs)


# Free local embeddings
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2-preview",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

#embedding_model = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma-db"
)    

# Query
results = vectorstore.similarity_search("vector database", k=2)
for r in results:
    print(r.page_content, r.metadata)