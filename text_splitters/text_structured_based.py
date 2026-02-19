from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(r"C:\Bluepixel\RAG_Langchain\Document_Loaders\policy.pdf")

docs=loader.load()

spliters = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap=0,
)

chunk=spliters.split_documents(docs)
print(len(chunk))
print(chunk[0])