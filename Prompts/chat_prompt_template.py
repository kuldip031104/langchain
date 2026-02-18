from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ('system',"you  are helful {domain}expert"),
    ('human',"Explin in simple terms , what is {topic}")
    
])
prompt = chat_template.invoke({'domain':'cricket','topic':'dusra'})

print(prompt)