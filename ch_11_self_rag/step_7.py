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
    retriever,
    route_after_decide,
)
from ch_11_self_rag.step_2 import is_relevant_node
from ch_11_self_rag.step_3 import (
    generate_from_context,
    no_relevant_docs_node,
    route_after_relevance,
)
from ch_11_self_rag.step_4 import is_supported_node
from ch_11_self_rag.step_5 import (
    MAX_RETRIES,
    revise_answer_node,
    route_after_is_supported,
)
from ch_11_self_rag.step_6 import is_useful_node
from utils.utils import get_logger

logger = get_logger()

logger.info("Self RAG step_7 execution started.")


class State(TypedDict):
    question: str
    answer: str

    # rewritter query if answer is not useful
    rewritten_query: str
    rewrite_tries: int

    need_retrieval: bool
    docs: list[Document]
    relevant_docs: list[Document]
    context: str

    is_supported: Literal["fully_supported", "partially_supported", "not_supported"]
    evidence: list[str]

    retries: int

    is_useful: Literal["useful", "not_useful"]
    useful_reason: str


# new retrieval node which retrieve docs based on question or rewritten query
def retrieve_node(state: State) -> State:
    q = state.get("question") or state.get("rewritter_query")
    return {"docs": retriever.invoke(input=q)}


# schema for rewrite decision
class RewriteDecision(BaseModel):
    rewritten_query: str = Field(
        description="Rewritten query optimized for vector retrieval against internal company PDFs"
    )


rewrite_for_retrieval_prompt = ChatPromptTemplate(
    messages=[
        (
            "system",
            (
                "Rewrite the user's QUESTION into a query optimized for vector retrieval over INTERNAL company PDFs.\n\n"
                "Rules:\n"
                "- Keep it short (6–16 words).\n"
                "- Preserve key entities (e.g., NexaAI, plan names).\n"
                "- Add 2–5 high-signal keywords that likely appear in policy/pricing docs.\n"
                "- Remove filler words.\n"
                "- Do NOT answer the question.\n"
                "- Output JSON with key: retrieval_query\n\n"
                "Examples:\n"
                "Q: 'Do NexaAI plans include a free trial?'\n"
                "-> {{'retrieval_query': 'NexaAI free trial duration trial period plans'}}\n\n"
                "Q: 'What is NexaAI refund policy?'\n"
                "-> {{'retrieval_query': 'NexaAI refund policy cancellation refund timeline charges'}}"
            ),
        ),
        (
            "human",
            "Question:\n{question}\nPrevious retrieval query:\n{retrieval_query}\nAnswer (if any):\n{answer}",
        ),
    ]
)

rewrite_llm = llm.with_structured_output(schema=RewriteDecision)


# rewrite question node
def rewrite_question_node(state: State) -> State:
    logger.info("Rewriting the question.")
    decision: RewriteDecision = rewrite_llm.invoke(
        rewrite_for_retrieval_prompt.format_messages(
            question=state.get("question"),
            retrieval_query=state.get("rewritten_query", ""),
            answer=state.get("answer"),
        )
    )
    return {
        "rewritten_query": decision.rewritten_query,
        "rewrite_tries": state.get("rewrite_tries", 0) + 1,
        "docs": [],  # optional reset docs, relevant docs and context so next iteration is clean
        "relevant_docs": [],
        "context": "",
    }


# updated route after is_useful
def route_after_is_useful(state: State) -> RouteVerdict:
    if state.get("is_useful") == "useful":
        return RouteVerdict.END
    if state.get("rewrite_tries", 0) >= MAX_RETRIES:
        return RouteVerdict.NO_ANSWER_FOUND
    return RouteVerdict.REWRITE_QUESTION


# define graph
graph = StateGraph(state_schema=State)

# add nodes
graph.add_node(Node.DECIDE_RETRIEVAL, decide_retrieval_node)
graph.add_node(Node.GENERATE_DIRECT, generate_direct_node)
graph.add_node(Node.RETRIEVE, retrieve_node)
graph.add_node(Node.IS_RELEVANT, is_relevant_node)
graph.add_node(Node.GENERATE_FROM_CONTEXT, generate_from_context)
graph.add_node(Node.NO_RELEVANT_DOCS, no_relevant_docs_node)
graph.add_node(Node.IS_SUPPORTED, is_supported_node)
graph.add_node(Node.REVISE_ANSWER, revise_answer_node)
graph.add_node(Node.IS_USEFUL, is_useful_node)
graph.add_node(Node.REWRITE_QUESTION, rewrite_question_node)

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
        RouteVerdict.ACCEPT_ANSWER: Node.IS_USEFUL,
        RouteVerdict.REVISE_ANSWER: Node.REVISE_ANSWER,
    },
)
graph.add_edge(
    Node.REVISE_ANSWER, Node.IS_SUPPORTED
)  # if revise loop back to is supported
graph.add_conditional_edges(
    source=Node.IS_USEFUL,
    path=route_after_is_useful,
    path_map={
        RouteVerdict.END: END,
        RouteVerdict.REWRITE_QUESTION: Node.REWRITE_QUESTION,
        RouteVerdict.NO_ANSWER_FOUND: Node.NO_RELEVANT_DOCS,
    },
)
graph.add_edge(Node.REWRITE_QUESTION, Node.RETRIEVE)

# compile graph
app = graph.compile()

if __name__ == "__main__":
    # Generates and writes the graph image directly to your project root / graphs
    png_data = app.get_graph().draw_mermaid_png()
    Path("graphs/self_rag_7.png").write_bytes(png_data)
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
