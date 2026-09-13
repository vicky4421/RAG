import time
from pathlib import Path
from typing import TypedDict

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from utils.logger import get_logger
from utils.utils import get_chroma_vector_store, get_llm, load_hf_embedding_model

logger = get_logger()

llm = get_llm()

embedding = load_hf_embedding_model(
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
)

vector_store = get_chroma_vector_store(
    collection_name="self_rag_demo",
    embedding_func=embedding,
    db_directory_name="chroma_db",
)

retriever = vector_store.as_retriever(search_kwargs={"k": 4})


def ingest_documents():
    # Avoid duplicate indexing if already populated
    count = vector_store._collection.count()
    if count > 0:
        logger.info(f"Collection already contains {count} chunks. Skipping ingestion.")
        return

    # load documents
    logger.info("Loading docs.")
    docs = (
        PyPDFLoader("./documents/Company_Policies.pdf").load()
        + PyPDFLoader("./documents/Company_Profile.pdf").load()
        + PyPDFLoader("./documents/Product_and_Pricing.pdf").load()
    )

    # split docs
    logger.info("Splitting docs")
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=150
    ).split_documents(documents=docs)

    # add documents to vector
    vector_store.add_documents(documents=chunks)


# define graph state
class State(TypedDict):
    question: str
    need_retrival: bool
    docs: list[Document]
    answer: str


class RetrieveDecision(BaseModel):
    should_retrieve: bool = Field(
        description="True if external documents are needed to answer reliebly, else False"
    )


if __name__ == "__main__":
    ingest_documents()

    # Generates and writes the graph image directly to your project root
    png_data = app.get_graph().draw_mermaid_png()
    Path("self_rag_1.png").write_bytes(png_data)
    print("Graph saved to root directory")
