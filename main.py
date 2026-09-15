from dotenv import load_dotenv
from langchain_mistralai import  ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

template = ChatPromptTemplate.from_messages([
    ("system", "You are a AI that summarize the text"),
    ("human", "{data}")
])

