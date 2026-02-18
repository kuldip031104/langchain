from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key="AIzaSyDpiemZryDYiJs8EtPLBZT4mt1ogos-kBw")
user_input= str(input("Enter the text:"))

template = PromptTemplate(
    template="""
    give me anser in single line.
     Question :{user_input}   
     """,
    input_variables=['user_input']
)
prompt=template.invoke(
    {
        'user_input':user_input
    }
)

chain = template | model
result=chain.invoke(prompt)
print(result.content) 