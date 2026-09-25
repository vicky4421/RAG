import logging
import operator
from pathlib import Path
from typing import Annotated

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, MessagesState, StateGraph

from ch_12_agentic_rag.nodes import Node, RouteVerdict
from ch_12_agentic_rag.step_0 import ingest_documents, llm
from ch_12_agentic_rag.step_1 import decide_retrieval_node
from ch_12_agentic_rag.step_5 import (
    agent_node,
    collect_tool_output_node,
    decide_decomposition_node,
    decompose_query_node,
    evaluate_docs_node,
    increment_tool_call_count_node,
    rewrite_query_node,
    route_after_agent_decision,
    route_after_decomposition_check,
    route_after_evaluation,
    route_after_rewrite_decision,
    route_after_tool_call_limit_check,
    tool_node,
)
from utils.logger import get_logger

# silent gemini warnings
logging.getLogger(name="google_genai.models").setLevel(level=logging.ERROR)

logger = get_logger()

logger.info("Agentic RAG step_6 execution started.")


# define state
class State(MessagesState):
    query: str
    response: str
    retrieved_docs: Annotated[list[Document], operator.add]
    relevant_docs: list[Document]
    context: str
    constructed_prompt: str
    needs_retrieval: bool
    tool_call_count: int
    max_tool_call_count: int
    is_relevant: bool
    rewritten_query: str
    rewrite_count: int
    needs_decomposition: bool


# rebuild prompt node
def build_prompt_node(state: State) -> State:
    logger.info("Rebuilding prompt.")

    rewrite_count = state.get("rewrite_count", 0)
    is_relevant = state.get("is_relevant", False)

    if rewrite_count >= 3 and not is_relevant:
        return {"constructed_prompt": "FALLBACK"}

    sections = ["You're a knowledgeble assistant."]

    if state.get("needs_decomposition"):
        sections.append(
            "The query was complex and split into subqueries for retrieval."
            "Synthesize a single coherent answer covering all parts."
        )

    if state.get("rewritten_query"):
        sections.append(
            "The original query was reforumulated during retrieval to improve result quality."
        )

    context = state.get("context") or ""
    relevant_docs = state.get("relevant_docs") or []

    if context:
        web_docs = [
            d for d in relevant_docs if d.metadata.get("source", "").startswith("http")
        ]
        pdf_docs = [
            d
            for d in relevant_docs
            if not d.metadata.get("source", "").startswith("http")
        ]
        if web_docs and pdf_docs:
            sections.append(
                "Answer using the following context from the internal documents and web search."
            )
        elif web_docs:
            sections.append("Answer using the following context from the web search.")
        else:
            sections.append(
                "Answer using the following context from the internal documents."
            )
        sections.append(f"Context:\n{context}")
        if web_docs:
            urls = "\n".join(f" - {d.metadata.get('source')}" for d in web_docs)
            sections.append(f"Sources:\n{urls}")
    else:
        sections.append(
            "No relevant documents were found. Answer from your general knowledge if possible."
        )

    return {"constructed_prompt": "\n\n".join(sections)}


# generate node
def generate_node(state: State) -> State:
    logger.info("Generating response")

    if state.get("constructed_prompt") == "FALLBACK":
        return {
            "response": (
                "I was unable to find relevant information to answer the query after multiple retrieval attempts."
                "The knowledge source does not appear to contain content that addresses the query."
            )
        }

    prompt = ChatPromptTemplate.from_messages(
        messages=[("system", "{constructed_prompt}"), ("user", "{query}")]
    )
    response = (prompt | llm).invoke(
        {
            "constructed_prompt": state.get("constructed_prompt"),
            "query": state.get("query"),
        }
    )
    return {"response": response.content}


# route after retrieval decision
def route_after_retrieval_decision(state: State) -> RouteVerdict:
    return (
        RouteVerdict.RETRIEVAL_NEEDED
        if state.get("needs_retrieval")
        else RouteVerdict.BUILD_PROMPT
    )


# build graph
graph = StateGraph(state_schema=State)

# add nodes
graph.add_node(Node.DECIDE_RETRIEVAL, decide_retrieval_node)
graph.add_node(Node.DECIDE_DECOMPOSITION, decide_decomposition_node)
graph.add_node(Node.DECOMPOSE, decompose_query_node)
graph.add_node(Node.AGENT, agent_node)
graph.add_node(Node.TOOL, tool_node)
graph.add_node(Node.INCREMENT_TOOL_CALL, increment_tool_call_count_node)
graph.add_node(Node.TOOL_OUTPUT, collect_tool_output_node)
graph.add_node(Node.EVALUATE_DOCS, evaluate_docs_node)
graph.add_node(Node.REWRITE_QUERY, rewrite_query_node)
graph.add_node(Node.BUILD_PROMPT, build_prompt_node)
graph.add_node(Node.GENERATE_RESPONSE, generate_node)

# add edges
graph.add_edge(START, Node.DECIDE_RETRIEVAL)
graph.add_conditional_edges(
    source=Node.DECIDE_RETRIEVAL,
    path=route_after_retrieval_decision,
    path_map={
        RouteVerdict.RETRIEVAL_NEEDED: Node.DECIDE_DECOMPOSITION,
        RouteVerdict.RETRIEVAL_NOT_NEEDED: Node.BUILD_PROMPT,
    },
)
graph.add_edge(Node.GENERATE_RESPONSE, END)
graph.add_conditional_edges(
    source=Node.DECIDE_DECOMPOSITION,
    path=route_after_decomposition_check,
    path_map={
        RouteVerdict.DECOMPOSITION_NOT_NEEDED: Node.AGENT,
        RouteVerdict.DECOMPOSITION_NEEDED: Node.DECOMPOSE,
    },
)
graph.add_edge(Node.DECOMPOSE, Node.AGENT)
graph.add_conditional_edges(
    source=Node.AGENT,
    path=route_after_agent_decision,
    path_map={
        RouteVerdict.TOOLS_REQUIRED: Node.TOOL,
        RouteVerdict.TOOLS_COMPLETE: Node.TOOL_OUTPUT,
    },
)
graph.add_edge(Node.TOOL, Node.INCREMENT_TOOL_CALL)
graph.add_conditional_edges(
    source=Node.INCREMENT_TOOL_CALL,
    path=route_after_tool_call_limit_check,
    path_map={
        RouteVerdict.TOOL_CALL_LIMIT_REACHED: Node.TOOL_OUTPUT,
        RouteVerdict.TOOL_CALL_WITHIN_LIMIT: Node.AGENT,
    },
)
graph.add_edge(Node.TOOL_OUTPUT, Node.EVALUATE_DOCS)
graph.add_conditional_edges(
    source=Node.EVALUATE_DOCS,
    path=route_after_evaluation,
    path_map={
        RouteVerdict.DOCS_RELEVANT: Node.BUILD_PROMPT,
        RouteVerdict.DOCS_NOT_RELEVANT: Node.REWRITE_QUERY,
    },
)
graph.add_edge(Node.BUILD_PROMPT, Node.GENERATE_RESPONSE)
graph.add_conditional_edges(
    source=Node.REWRITE_QUERY,
    path=route_after_rewrite_decision,
    path_map={
        RouteVerdict.REWRITE_EXHAUSTED: Node.BUILD_PROMPT,
        RouteVerdict.REWRITE_NEEDED: Node.AGENT,
    },
)

# compile graph
app = graph.compile()

# compile graph
app = graph.compile()

if __name__ == "__main__":
    ingest_documents()

    # Generates and writes the graph image directly to your project root / graphs
    png_data = app.get_graph().draw_mermaid_png()
    Path("graphs/agentic_rag_6.png").write_bytes(png_data)
    logger.info("Graph saved to root directory")

    result: State = app.invoke(
        {
            "query": "According to the report, how much battery pack costs fallen over the past decade, and what is the current price of oil in USD for brent crude?",
            "messages": [],
            "context": "",
            "retrieved_docs": [],
            "max_tool_call_count": 5,
            "rewrite_count": 0,
        }
    )

    logger.info(f"Is relevant: {result.get('is_relevant')}")
    logger.info(f"Retrieval needed: {result.get('needs_retrieval')}")
    if result.get("retrieved_docs"):
        logger.info(f"Length of retrieved docs: {len(result.get('retrieved_docs'))}")
    else:
        logger.info("No docs retrieved.")
    logger.info(f"No. of tool calls: {result.get('tool_call_count')}")
    logger.info(f"Rewrite counts: {result.get('rewrite_count')}")
    logger.info(f"Query: {result.get('query')}")
    logger.info(f"Constructed prompt: {result.get('constructed_prompt')}")
    logger.info(f"Response:\n {result.get('response')[0]['text']}")
