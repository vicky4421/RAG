import os

from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings

from utils.logger import get_logger

logger = get_logger()


def initiate_llm(model: str = "google_genai:gemini-3.1-flash-lite"):
    try:
        llm = init_chat_model(model=model)
        logger.info(f"LLM initiated successfully: {model}")
        return llm
    except Exception as e:
        logger.critical(f"Unexpected error initializing llm: {e}")
        raise


def initiate_hf_embedding_model(
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
