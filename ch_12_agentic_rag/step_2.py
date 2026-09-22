# define state
import operator
from typing import Annotated

from langchain_core.documents import Document
from langchain_core.tools import tool
from langgraph.graph import MessagesState

from ch_12_agentic_rag.step_0 import vector_store
from utils.logger import get_logger

logger = get_logger()

logger.info("Agentic RAG step_2 execution started.")


class State(MessagesState):
    query: str
    retrieved_docs: Annotated[list[Document], operator.add]
    context: Annotated[str, operator.add]
    needs_retrieval: bool


@tool(response_format="content_and_artifact")
def vector_store_search(query: str, k: int = 3) -> tuple[str, list[Document]]:
    logger.info("Tool called: vector_store_search")

    """
    Search the vector store for relevant document passages.
    Adjust k (default 3) to retrieve more or fewer passages.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(input=query)

    context = "\n\nVector Store Results\n\n" + "\n\n".join(d.page_content for d in docs)

    # due to response format 'content_and_artifacts' we can return content=context and artifact=docs
    return context, docs

@tool(response_format="content_and_artifact")
def web_search(query: str, max_results: int = 3) -> tuple[str, list[Document]]:
    logger.info('Tool called: web search')
    '''
    Search the web for current or real time information.
    Adjust max_results (default 3) to control how many results are returned.
    '''
    client = 