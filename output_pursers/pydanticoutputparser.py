from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel , Field

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class Person(BaseModel):
    name: str = Field(description="name of Person")
    age : int = Field(gt=18,description='Age of person')
    city : str = Field(description='Name of city person belongs to')
    
    
parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate details of a fictional person.\n{format_instruction}",
    input_variables=[],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    }
)

chain = template | model | parser

final_result = chain.invoke({'place':'sri lankan'})

print(final_result)