from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

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

# First generation
prompt1 = template1.format(topic="black hole")
result1 = model.invoke(prompt1)

# Second generation
prompt2 = template2.format(text=result1.content)
result2 = model.invoke(prompt2)

print(result2)
