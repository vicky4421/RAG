from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from rich.console import Console

from utils.logger import get_logger
from utils.utils import initiate_hf_embedding_model, initiate_llm

load_dotenv()
console = Console()
logger = get_logger()

# chroma db path (move to root in next chapter)
chroma_path = Path(__file__).resolve().parent / "chroma_db"

logger.info(msg="Pipeline initiated.")

# llm = init_chat_model("google_genai:gemini-3.1-flash-lite")
llm = initiate_llm()

embeddings = initiate_hf_embedding_model(
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
)

vector_store = Chroma(
    collection_name="corrective_rag_demo",
    embedding_function=embeddings,
    persist_directory=str(chroma_path),
)
logger.info("Vector store initiated successfully.")

retriver = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
logger.info("Retriever initiated successfully.")


# define state schema
class State(TypedDict):
    question: str
    docs: list[Document]
    answer: str

    strips: list[str]  # output of decomposition (sentence strips)
    kept_strips: list[str]  # after filtering (kept sentences)
    refined_context: str  # recomposed internal knowledge (joined kept_strips)
