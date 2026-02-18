from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

from typing import TypedDict

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

class Review (TypedDict):
    summary :str
    sentiment : str
    
struchured_model =model.with_structured_output(Review)
result = model.invoke("""
                      The most striking aspect of a Vishal Bhardwaj film, even one that does not fully land, is that there is nary a dull moment in it. He does not hold back. He stays within the parameters of Mumbai's mainstream practices while seeking ways around and beyond the commercial movie template.

                      """)

print(result)
print(result['summary'])
print(result['sentiment'])