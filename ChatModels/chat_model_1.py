from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv
# load_dotenv(dotenv_path=r"C:\Bluepixel\RAG_Langchain\.env")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key="AIzaSyDpiemZryDYiJs8EtPLBZT4mt1ogos-kBw")

result=model.invoke("what is the capital of india?")
print(result.content) 