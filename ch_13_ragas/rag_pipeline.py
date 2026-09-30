from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.logger import get_logger
from utils.utils import get_chroma_vector_store, get_llm, load_hf_embedding_model

logger = get_logger()

llm = get_llm()

embedding = load_hf_embedding_model(
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
)

vector_store = get_chroma_vector_store(
    collection_name="rag_eval",
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

    # Points directly to D:\AI\RAG\11_self_rag\documents
    DOCS_DIR = Path(__file__).resolve().parent / "documents"

    docs = PyPDFLoader(str(DOCS_DIR / "sustainable_developement.pdf")).load()

    # split docs
    logger.info("Splitting docs")
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150
    ).split_documents(documents=docs)

    # add documents to vector
    vector_store.add_documents(documents=chunks)


prompt = ChatPromptTemplate.from_template(
    "Use only the context below to answer the question. "
    "If the context does not contain the answer, say you don't know.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)

chain = prompt | llm
