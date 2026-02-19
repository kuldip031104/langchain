from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

parser = StrOutputParser()

prompt = PromptTemplate(
    template='Write a summarey for this text -n \n {text}',
    input_variables=['text']
)

loader = TextLoader('ai.txt')

docs=loader.load()

print(docs)

chain = prompt |model |parser

print(chain.invoke({'text':docs[0].page_content}))