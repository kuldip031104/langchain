# ==============================
# 1️⃣  IMPORTS
# ==============================

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os


# ==============================
# 2️⃣  LOAD ENV VARIABLES
# ==============================

load_dotenv()


# ==============================
# 3️⃣  GET YOUTUBE TRANSCRIPT
# ==============================

def get_transcript(video_id: str) -> str:
    try:
        api = YouTubeTranscriptApi()
        transcript_obj = api.fetch(video_id)
        transcript_list = transcript_obj.to_raw_data()
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        return transcript

    except TranscriptsDisabled:
        print("No captions available for this video.")
        return ""


# ==============================
# 4️⃣  SPLIT DOCUMENT
# ==============================

def split_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=150
    )
    return splitter.create_documents([text])


# ==============================
# 5️⃣  CREATE VECTOR STORE
# ==============================

def create_vector_store(documents):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store


# ==============================
# 6️⃣  CREATE RAG CHAIN
# ==============================

def create_rag_chain(vector_store):

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 8}
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )

    prompt = PromptTemplate(
        template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

Context:
{context}

Question: {question}
""",
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )

    parser = StrOutputParser()

    rag_chain = parallel_chain | prompt | llm | parser

    return rag_chain


# ==============================
# 7️⃣  MAIN EXECUTION
# ==============================

if __name__ == "__main__":

    video_id = "Gfr50f6ZBvo"

    # Step 1: Get transcript
    transcript = get_transcript(video_id)

    # Step 2: Split
    documents = split_text(transcript)

    # Step 3: Create Vector Store
    vector_store = create_vector_store(documents)

    # Step 4: Create RAG Chain
    rag_chain = create_rag_chain(vector_store)

    # Step 5: Ask Question
    question = "who is the main charactor in this video ?"

    response = rag_chain.invoke(question)
    print("\nAnswer:\n")
    print(response)
