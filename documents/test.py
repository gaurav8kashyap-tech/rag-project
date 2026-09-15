from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import CharacterTextSplitter


splitter = CharacterTextSplitter(
    separator = "",
    chunk_size=10, 
    chunk_overlap=1
)

data = TextLoader("documents/text.txt")
docs = data.load()

chunks = splitter.split_documents(docs)

#print(chunks)

for ch in chunks:
    print(ch.page_content)

    print("---  ---")
    print(" ")
    print("")