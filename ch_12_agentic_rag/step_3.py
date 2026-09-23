import logging
import operator
from pathlib import Path
from typing import Annotated

from langchain_core.documents import Document
from langgraph.graph import END, START, MessagesState, StateGraph

from ch_12_agentic_rag.nodes import Node, RouteVerdict
from ch_12_agentic_rag.step_0 import ingest_documents
from ch_12_agentic_rag.step_1 import (
    decide_retrieval_node,
    generate_node,
    route_after_retrieval_decision,
)
from ch_12_agentic_rag.step_2 import (
    agent_node,
    collect_tool_output_node,
    route_after_agent_decision,
    tool_node,
)
from utils.logger import get_logger

# silent gemini warnings
logging.getLogger(name="google_genai.models").setLevel(level=logging.ERROR)

logger = get_logger()

logger.info("Agentic RAG step_3 execution started.")


class State(MessagesState):
    query: str
    response: str
    retrieved_docs: Annotated[list[Document], operator.add]
    needs_retrieval: bool
    context: Annotated[str, operator.add]
    tool_call_count: int  # increment after each tool call
    max_tool_call_count: int  # cap on tool calls


# retrieval limit node
def increment_tool_call_count_node(state: State) -> State:
    logger.info("Incrementing tool call count.")
    count = state.get("tool_call_count", 0)
    return {"tool_call_count": count + 1}


# route after tool call limit check
def route_after_tool_call_limit_check(state: State) -> RouteVerdict:
    return (
        RouteVerdict.TOOL_CALL_LIMIT_REACHED
        if state.get("tool_call_count") >= state.get("max_tool_call_count")
        else RouteVerdict.TOOLS_REQUIRED
    )


# define graph
graph = StateGraph(state_schema=State)

# add nodes
graph.add_node(Node.DECIDE_RETRIEVAL, decide_retrieval_node)
graph.add_node(Node.AGENT, agent_node)
graph.add_node(Node.TOOL, tool_node)
graph.add_node(Node.INCREMENT_TOOL_CALL, increment_tool_call_count_node)
graph.add_node(Node.TOOL_OUTPUT, collect_tool_output_node)
graph.add_node(Node.GENERATE_RESPONSE, generate_node)

# add edges
graph.add_edge(START, Node.DECIDE_RETRIEVAL)
graph.add_conditional_edges(
    source=Node.DECIDE_RETRIEVAL,
    path=route_after_retrieval_decision,
    path_map={
        RouteVerdict.NEED_RETRIEVAL: Node.AGENT,
        RouteVerdict.GENERATE_DIRECT: Node.GENERATE_RESPONSE,
    },
)
graph.add_edge(Node.GENERATE_RESPONSE, END)
graph.add_conditional_edges(
    source=Node.AGENT,
    path=route_after_agent_decision,
    path_map={
        RouteVerdict.TOOLS_REQUIRED: Node.TOOL,
        RouteVerdict.TOOLS_COMPLETE: Node.TOOL_OUTPUT,
    },
)
graph.add_edge(Node.TOOL_OUTPUT, Node.GENERATE_RESPONSE)
graph.add_edge(Node.TOOL, Node.INCREMENT_TOOL_CALL)
graph.add_conditional_edges(
    source=Node.INCREMENT_TOOL_CALL,
    path=route_after_tool_call_limit_check,
    path_map={
        RouteVerdict.TOOL_CALL_LIMIT_REACHED: Node.TOOL_OUTPUT,
        RouteVerdict.TOOLS_REQUIRED: Node.AGENT,
    },
)

# compile graph
app = graph.compile()

if __name__ == "__main__":
    ingest_documents()

    # Generates and writes the graph image directly to your project root / graphs
    png_data = app.get_graph().draw_mermaid_png()
    Path("graphs/agentic_rag_3.png").write_bytes(png_data)
    logger.info("Graph saved to root directory")

    result: State = app.invoke(
        {
            "query": "Compare what the EV report projects for 2030 oil demand with current oil demand due to US-Iran war.",
            "messages": [],
            "context": "",
            "retrieved_docs": [],
            "max_tool_call_count": 5,
        }
    )

    logger.info(f"Retrieval needed: {result.get('needs_retrieval')}")
    if result.get("retrieved_docs"):
        logger.info(f"Length of retrieved docs: {len(result.get('retrieved_docs'))}")
    else:
        logger.info("No docs retrieved.")
    logger.info(f"No. of tool calls: {result.get('tool_call_count')}")
    logger.info(f"Query: {result.get('query')}")
    logger.info(f"Response: {result.get('response').content[0]['text']}")
