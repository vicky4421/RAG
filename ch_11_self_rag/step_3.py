from pathlib import Path
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from ch_11_self_rag.nodes import Node, RouteVerdict
from ch_11_self_rag.step_1 import (
    decide_retrieval_node,
    generate_direct_node,
    llm,
    retrieve_node,
    route_after_decide,
)
from ch_11_self_rag.step_2 import is_relevant_node
from utils.logger import get_logger

logger = get_logger()

logger.info("Self RAG step_3 execution started.")


# define state
class State(TypedDict):
    question: str
    need_retrieval: bool
    docs: list[Document]
    relevant_docs: list[Document]
    context: str  # to merge relevant docs
    answer: str


# prompt for generate from context
generate_from_context = ChatPromptTemplate.from_messages(
    messages=[
        (
            "system",
            """
                You're a business RAG assistant.\n
                Answer the user's question using ONLY the provided context.\n
                If the context doesn't contain enough information, say: \n
                No relevant documents found.
                Do not use outside knowledge
            """,
        ),
        ("human", "Question:\n{question}Context:\n{context}"),
    ]
)


def generate_from_context_node(state: State) -> State:
    logger.info("Generating from context.")

    # merge relevant docs
    context = "\n\n---\n\n".join(
        [d.page_content for d in state.get("relevant_docs", "")]
    ).strip()

    if not context:
        return state["answer":"No relevant documents found", "context":""]

    out = llm.invoke(
        generate_from_context.format_messages(
            question=state["question"], context=context
        )
    )
    return {"answer": out.content, "context": context}


def no_relevant_docs_node(state: State) -> State:
    return {"answer": "No relevant docs found!", "context": ""}


def route_after_relevance(state: State) -> RouteVerdict:
    if state.get("relevant_docs") and len(state.get("relevant_docs")) > 0:
        return RouteVerdict.GENERATE_FROM_CONTEXT
    return RouteVerdict.NO_RELEVANT_DOCS


# create graph
graph = StateGraph(state_schema=State)

# add nodes
graph.add_node(Node.DECIDE_RETRIEVAL, decide_retrieval_node)
graph.add_node(Node.GENERATE_DIRECT, generate_direct_node)
graph.add_node(Node.RETRIEVE, retrieve_node)
graph.add_node(Node.IS_RELEVANT, is_relevant_node)
graph.add_node(Node.GENERATE_FROM_CONTEXT, generate_from_context_node)
graph.add_node(Node.NO_RELEVANT_DOCS, no_relevant_docs_node)

# add edges
graph.add_edge(START, Node.DECIDE_RETRIEVAL)
graph.add_conditional_edges(
    source=Node.DECIDE_RETRIEVAL,
    path=route_after_decide,
    path_map={
        RouteVerdict.GENERATE_DIRECT: Node.GENERATE_DIRECT,
        RouteVerdict.RETRIEVE: Node.RETRIEVE,
    },
)
graph.add_edge(Node.GENERATE_DIRECT, END)
graph.add_edge(Node.RETRIEVE, Node.IS_RELEVANT)
graph.add_conditional_edges(
    source=Node.IS_RELEVANT,
    path=route_after_relevance,
    path_map={
        RouteVerdict.GENERATE_FROM_CONTEXT: Node.GENERATE_FROM_CONTEXT,
        RouteVerdict.NO_RELEVANT_DOCS: Node.NO_RELEVANT_DOCS,
    },
)
graph.add_edge(Node.GENERATE_FROM_CONTEXT, END)
graph.add_edge(Node.NO_RELEVANT_DOCS, END)

# compile graph
app = graph.compile()

if __name__ == "__main__":
    # Generates and writes the graph image directly to your project root / graphs
    png_data = app.get_graph().draw_mermaid_png()
    Path("graphs/self_rag_3.png").write_bytes(png_data)
    print("Graph saved to root directory")

    result: State = app.invoke({"question": "What is refund policy of Nexa AI?"})

    logger.info(f"Need retrieval? : {result['need_retrieval']}")

    logger.info(f"Qustion: {result.get('question')}")

    if "answer" in result and result.get("context") == "":
        logger.info(f"{result['answer']}")
    else:
        logger.info(f"{result['answer'][0]['text']}")
