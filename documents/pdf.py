from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter

data = PyPDFLoader("documents/deeplearningbyOrelly.pdf")

docs = data.load()
#print(docs);
# token based text splitter 
splitter = TokenTextSplitter(
    chunk_size=1000, 
    chunk_overlap=10
)

chunks = splitter.split_documents(docs)
print(chunks[2].page_content)