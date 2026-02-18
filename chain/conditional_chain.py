from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel ,Field
from typing import Literal 
load_dotenv()

model = ChatGoogleGenerativeAI(model ='gemini-2.5-flash')

parser1 = StrOutputParser()

class Feedback(BaseModel):
    
    sentiment : Literal[ 'positive','nagative'] = Field (description='Give the sentiment of feedback')
    
parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='classify the sentiment of the following feed text into positive or nagative\n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

classifier_chain = prompt1 |model |parser2

prompt2 =PromptTemplate(
    template='write an appopiate response to this positive feedback\n {feedback}',
    input_variables=['feedback']
)

prompt3 =PromptTemplate(
    template='write an appopiate response to this nagative feedback\n {feedback}',
    input_variables=['feedback']
)
brach_chain =RunnableBranch(
    (lambda x:x.sentiment=='positive',prompt2|model|parser1),
    (lambda x:x.sentiment=='nagative',prompt3|model|parser1),
    RunnableLambda(lambda x:'could not find sentiment')
)

chain = classifier_chain | brach_chain

print(chain.invoke({'feedback':' this is good smartphone'}))


