from dotenv import load_dotenv
from langchain_mistralai import  ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader


load_dotenv()

data = PyPDFLoader("documents/pythoninterviewquestion.pdf")
#data = TextLoader("documents/text.txt")
docs = data.load()


template = ChatPromptTemplate.from_messages([
    ("system", "You are a AI that summarize the text"),
    ("human", "{data}")
])

model = ChatMistralAI(model = "mistral-small-latest")

prompt = template.format_messages(data = docs[2].page_content)
result = model.invoke(prompt)
print(result.content)
