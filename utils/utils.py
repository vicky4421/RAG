import inspect
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from tavily import TavilyClient

from utils.logger import get_logger

load_dotenv()
logger = get_logger()


def get_llm(model: str = "google_genai:gemini-3.1-flash-lite", temp: float = 0.0):
    try:
        llm = init_chat_model(model=model, temperature=temp)
        logger.info(f"LLM initiated successfully: {model}")
        return llm
    except Exception as e:
        logger.critical(f"Unexpected error initializing llm: {e}")
        raise


def get_agent_llm(model: str = "google_genai:gemini-3.1-flash-lite", temp: float = 0.0):
    try:
        llm = init_chat_model(model=model, temperature=temp)
        logger.info(f"Agent LLM initiated successfully: {model}")
        return llm
    except Exception as e:
        logger.critical(f"Unexpected error initializing llm: {e}")
        raise


def load_hf_embedding_model(
    *,
    model_kwargs: dict | None = None,
    encode_kwargs: dict | None = None,
    model_name: str | None = None,
    cache_folder: str | None = None,
) -> HuggingFaceEmbeddings:
    emb_model = os.getenv("EMBEDDING_MODEL") or model_name
    cache_dir = os.getenv("HUGGINGFACE_EMBEDDING_MODEL_DIRECTORY") or cache_folder

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=emb_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            cache_folder=cache_dir,
        )
        logger.info(f"Embedding model initiated successfully: {emb_model}")
        return embeddings
    except Exception as e:
        logger.critical(f"Unexpected error initializing embedding model: {e}")
        raise


def get_chroma_vector_store(
    *,
    collection_name: str,
    embedding_func: HuggingFaceEmbeddings,
    db_directory_name: str,
) -> Chroma:

    # Frame 1 is the function calling get_chroma_vector_store
    caller_frame = inspect.stack()[1]
    caller_file = caller_frame.filename

    chroma_path = Path(caller_file).resolve().parent / db_directory_name
    logger.info(f"Resolved caller directory Chroma path: {chroma_path}")

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_func,
        persist_directory=str(chroma_path),
    )

    logger.info(f"Vector store '{collection_name}' initialized successfully.")
    return vector_store


def get_tavily_client() -> TavilyClient:
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        logger.info("Tavily Client initiated successfully.")
        return client
    except Exception as e:
        logger.critical(f"Unexpected error initializing tavily client: {e}")
        raise
