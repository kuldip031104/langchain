from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(r"C:\Bluepixel\RAG_Langchain\Document_Loaders\policy.pdf")

docs=loader.load()

spliters = CharacterTextSplitter(
    chunk_size = 30,
    chunk_overlap=0,
    separator=''
)

result=spliters.split_documents(docs)

print(result[0])