from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Use only HuggingFaceEndpoint
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Write a 5 line summary on the following report:\n{text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = template1  | model | parser  | template2 | model | parser

result=chain.invoke({'topic':'black hole'})

print(result)