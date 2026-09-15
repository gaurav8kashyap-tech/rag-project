from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from line_profiler import LineProfiler

import os
load_dotenv()

from langchain_core.documents import Document

docs = [
    Document(page_content="Python is widely used in Artificial Inteligence.", metadata={"source": "doc1"}),
    Document(page_content="Pandas is used for data manipulation and analysis in python.", metadata={"source": "doc2"}),
    Document(page_content="Neural networks are a key component of deep learning.", metadata={"source": "doc3"})
]
# Free local embeddings
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2-preview",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

#embedding_model = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma-db"
)    

# Query
results = vectorstore.similarity_search("What is used for data analysis in Python?", k=2)
for r in results:
    print(r.page_content, r.metadata)
    LineProfiler(r.page_content, r.metadata)


retriver = vectorstore.as_retriever()

docs = retriver.invoke("Explain deep learning")

for d in docs:
    print(d.page_content)