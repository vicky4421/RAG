from pathlib import Path
from typing import Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from ch_11_self_rag.nodes import Node, RouteVerdict
from ch_11_self_rag.step_1 import (
    decide_retrieval_node,
    generate_direct_node,
    llm,
    retrieve_node,
)
from ch_11_self_rag.step_2 import is_relevant_node, route_after_decide
from ch_11_self_rag.step_3 import generate_from_context_node, no_relevant_docs_node
from ch_11_self_rag.step_4 import is_supported_node, route_after_relevance
from utils.logger import get_logger

logger = get_logger()

logger.info("Self RAG step_5 execution started.")


class State(TypedDict):
    question: str
    answer: str
    need_retrieval: bool
    docs: list[Document]
    relevant_docs: list[Document]
    context: str
    is_supported: Literal["fully_supported", "partially_supported", "not_supported"]
    evidence: list[str]
    retries: int


MAX_RETRIES = 10


# route after is_supported
def route_after_is_supported(state: State) -> RouteVerdict:
    # accept answer if fully supported
    if state.get("is_supported") == "fully_supported":
        return RouteVerdict.ACCEPT_ANSWER

    # if loop is exceeding max retries
    if state.get("retries", 0) >= MAX_RETRIES:
        return (
            RouteVerdict.ACCEPT_ANSWER
        )  # or we can add a new node which says not supported answer

    # otherwise revise
    return RouteVerdict.REVISE_ANSWER


def accept_answer_node(state: State) -> State:
    logger.info("Accepting answer after finding it supported.")
    # return nothing, keep answer as is
    return {}


revise_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a STRICT reviser.\n\n"
                "You must output based on the following format:\n\n"
                "FORMAT (quote-only answer):\n"
                "- <direct quote from the CONTEXT>\n"
                "- <direct quote from the CONTEXT>\n\n"
                "Rules:\n"
                "- Use ONLY the CONTEXT.\n"
                "- Do NOT add any new words besides bullet dashes and the quotes themselves.\n"
                "- Do NOT explain anything.\n"
                "- Do NOT say 'context', 'not mentioned', 'does not mention', 'not provided', etc.\n"
            ),
        ),
        (
            "human",
            "Question:\n{question}\n\nCurrent Answer:\n{answer}\n\nCONTEXT:\n{context}",
        ),
    ]
)


# revise answer node
def revise_answer_node(state: State) -> State:
    logger.info("Revising answer after finding it supported.")

    out = llm.invoke(
        revise_prompt.format_messages(
            question=state.get("question"),
            answer=state.get("answer", ""),
            context=state.get("context", ""),
        )
    )
    logger.info(f"Retries: {state.get('retries')}")
    return {"answer": out.content, "retries": state.get("retries", 0) + 1}


# build graph
graph = StateGraph(state_schema=State)

# add nodes
graph.add_node(Node.DECIDE_RETRIEVAL, decide_retrieval_node)
graph.add_node(Node.GENERATE_DIRECT, generate_direct_node)
graph.add_node(Node.RETRIEVE, retrieve_node)
graph.add_node(Node.IS_RELEVANT, is_relevant_node)
graph.add_node(Node.GENERATE_FROM_CONTEXT, generate_from_context_node)
graph.add_node(Node.NO_RELEVANT_DOCS, no_relevant_docs_node)
graph.add_node(Node.IS_SUPPORTED, is_supported_node)
graph.add_node(Node.ACCEPT_ANSWER, accept_answer_node)
graph.add_node(Node.REVISE_ANSWER, revise_answer_node)

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
graph.add_edge(Node.GENERATE_DIRECT, END)  # generate from its parametric knowledge
graph.add_edge(Node.RETRIEVE, Node.IS_RELEVANT)
graph.add_conditional_edges(
    source=Node.IS_RELEVANT,
    path=route_after_relevance,
    path_map={
        RouteVerdict.GENERATE_FROM_CONTEXT: Node.GENERATE_FROM_CONTEXT,
        RouteVerdict.NO_RELEVANT_DOCS: Node.NO_RELEVANT_DOCS,
    },
)
graph.add_edge(Node.NO_RELEVANT_DOCS, END)  # if no answer found
# verify -> accept / revise loop
graph.add_edge(Node.GENERATE_FROM_CONTEXT, Node.IS_SUPPORTED)
graph.add_conditional_edges(
    source=Node.IS_SUPPORTED,
    path=route_after_is_supported,
    path_map={
        RouteVerdict.ACCEPT_ANSWER: Node.ACCEPT_ANSWER,
        RouteVerdict.REVISE_ANSWER: Node.REVISE_ANSWER,
    },
)
graph.add_edge(Node.REVISE_ANSWER, Node.IS_SUPPORTED)  # loop back to verify again
graph.add_edge(Node.ACCEPT_ANSWER, END)

app = graph.compile()

if __name__ == "__main__":
    # Generates and writes the graph image directly to your project root / graphs
    png_data = app.get_graph().draw_mermaid_png()
    Path("graphs/self_rag_5.png").write_bytes(png_data)
    logger.info("Graph saved to root directory")

    result: State = app.invoke(
        {"question": "Describe NexaAI’s company culture."},
        config={"recursion_limit": 80},
    )

    logger.info(f"Need retrieval? : {result['need_retrieval']}")

    logger.info(f"Qustion: {result.get('question')}")

    if "answer" in result and result.get("context") == "":
        logger.info(f"{result['answer']}")
    else:
        logger.info(f"{result['answer'][0]['text']}")

    logger.info(f"Docs: {result.get('docs')}")
    logger.info(f"Context: {result.get('context')}")
    logger.info(f"Is supported: {result.get('is_supported')}")
    logger.info(f"Evidence: {result.get('evidence')}")
    logger.info(f"Retries: {result.get('retries')}")
