from pathlib import Path
from typing import Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from ch_11_self_rag.nodes import Node, RouteVerdict
from ch_11_self_rag.step_1 import (
    decide_retrieval_node,
    generate_direct_node,
    llm,
    retrieve_node,
    route_after_decide,
)
from ch_11_self_rag.step_2 import is_relevant_node
from ch_11_self_rag.step_3 import (
    generate_from_context_node,
    no_relevant_docs_node,
    route_after_relevance,
)
from ch_11_self_rag.step_4 import is_supported_node
from ch_11_self_rag.step_5 import revise_answer_node, route_after_is_supported
from utils.logger import get_logger

logger = get_logger()

logger.info("Self RAG step_6 execution started.")


# define state
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

    # usefulness check
    is_useful: Literal["useful", "not_useful"]
    useful_reason: str


class IsUsefulDecision(BaseModel):
    is_useful: Literal["useful", "not_useful"]
    reason: str = Field(description="Short reason in one line")


is_useful_prompt = ChatPromptTemplate.from_messages(
    messages=[
        (
            "system",
            (
                "You are judging USEFULNESS of the ANSWER for the QUESTION.\n\n"
                "Goal:\n"
                "- Decide if the answer actually addresses what the user asked.\n\n"
                "Return JSON with keys: isuse, reason.\n"
                "isuse must be one of: useful, not_useful.\n\n"
                "Rules:\n"
                "- useful: The answer directly answers the question or provides the requested specific info.\n"
                "- not_useful: The answer is generic, off-topic, or only gives related background without answering.\n"
                "- Do NOT use outside knowledge.\n"
                "- Do NOT re-check grounding (IsSUP already did that). Only check: 'Did we answer the question?'\n"
                "- Keep reason to 1 short line."
            ),
        ),
        ("human", "Question:\n{question}\n\nAnswer:\n{answer}"),
    ]
)

is_useful_llm = llm.with_structured_output(schema=IsUsefulDecision)


# is useful node
def is_useful_node(state: State) -> State:
    logger.info("Checking if answer is useful.")
    decison: IsUsefulDecision = is_useful_llm.invoke(
        is_useful_prompt.format_messages(
            question=state.get("question"), answer=state.get("answer", "")
        )
    )
    return {"is_useful": decison.is_useful, "useful_reason": decison.reason}


# route
def route_after_is_useful(state: State) -> RouteVerdict:
    if state.get("is_useful") == "useful":
        return RouteVerdict.END
    return RouteVerdict.NO_ANSWER_FOUND


# define graph
graph = StateGraph(state_schema=State)

# add nodes
graph.add_node(Node.DECIDE_RETRIEVAL, decide_retrieval_node)
graph.add_node(Node.GENERATE_DIRECT, generate_direct_node)
graph.add_node(Node.RETRIEVE, retrieve_node)
graph.add_node(Node.IS_RELEVANT, is_relevant_node)
graph.add_node(Node.GENERATE_FROM_CONTEXT, generate_from_context_node)
graph.add_node(Node.NO_RELEVANT_DOCS, no_relevant_docs_node)
graph.add_node(Node.IS_SUPPORTED, is_supported_node)
graph.add_node(Node.REVISE_ANSWER, revise_answer_node)
# is useful?
graph.add_node(Node.IS_USEFUL, is_useful_node)

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
graph.add_edge(Node.NO_RELEVANT_DOCS, END)
graph.add_edge(Node.GENERATE_FROM_CONTEXT, Node.IS_SUPPORTED)
graph.add_conditional_edges(
    source=Node.IS_SUPPORTED,
    path=route_after_is_supported,
    path_map={
        RouteVerdict.ACCEPT_ANSWER: Node.IS_USEFUL,  # after accepting check if useful (prev: blank accept answer node was defined)
        RouteVerdict.REVISE_ANSWER: Node.REVISE_ANSWER,
    },
)
graph.add_edge(
    Node.REVISE_ANSWER, Node.IS_SUPPORTED
)  # loop back revised answer to check if supported

# is useful routing
graph.add_conditional_edges(
    source=Node.IS_USEFUL,
    path=route_after_is_useful,
    path_map={
        RouteVerdict.END: END,
        RouteVerdict.NO_ANSWER_FOUND: Node.NO_RELEVANT_DOCS,
    },
)

# compile graph
app = graph.compile()

if __name__ == "__main__":
    # Generates and writes the graph image directly to your project root / graphs
    png_data = app.get_graph().draw_mermaid_png()
    Path("graphs/self_rag_6.png").write_bytes(png_data)
    logger.info("Graph saved to root directory")

    result: State = app.invoke(
        {"question": "What is refund policy of NexaAI"},
        config={"recursion_limit": 80},
    )

    logger.info(f"Need retrieval? : {result['need_retrieval']}")

    logger.info(f"Qustion: {result.get('question')}")

    if "answer" in result and result.get("context") == "":
        logger.info(f"{result['answer']}")
    else:
        logger.info(f"{result['answer'][0]['text']}")

    logger.info(f"Length of fetched documens: {len(result.get('docs'))}")
    logger.info(f"Context: {result.get('context')}")
    logger.info(f"Is supported: {result.get('is_supported')}")
    logger.info(f"Evidence: {result.get('evidence')}")
    logger.info(f"Retries: {result.get('retries')}")
    logger.info(f"Useful reason: {result.get('useful_reason')}")
