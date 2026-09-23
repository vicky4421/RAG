# define state
import logging
import operator
from pathlib import Path
from typing import Annotated

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from ch_12_agentic_rag.nodes import Node, RouteVerdict
from ch_12_agentic_rag.step_0 import ingest_documents, vector_store
from ch_12_agentic_rag.step_1 import (
    decide_retrieval_node,
    generate_node,
    route_after_retrieval_decision,
)
from utils.logger import get_logger
from utils.utils import get_agent_llm, get_tavily_client

# silent gemini warnings
logging.getLogger(name="google_genai.models").setLevel(level=logging.ERROR)

logger = get_logger()

logger.info("Agentic RAG step_2 execution started.")

tavily_client = get_tavily_client()
agent_llm = get_agent_llm()


class State(MessagesState):
    query: str
    response: str
    retrieved_docs: Annotated[list[Document], operator.add]
    context: Annotated[str, operator.add]
    needs_retrieval: bool


@tool(response_format="content_and_artifact")
def vector_store_search_tool(query: str, k: int = 3) -> tuple[str, list[Document]]:
    """
    Search the vector store for relevant document passages.
    Adjust k (default 3) to retrieve more or fewer passages.
    """
    logger.info("Tool called: vector_store_search")
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(input=query)

    context = "\n\n**Vector Store Results**\n\n" + "\n\n".join(
        d.page_content for d in docs
    )

    # due to response format 'content_and_artifacts' we can return content=context and artifact=docs
    return context, docs


@tool(response_format="content_and_artifact")
def web_search_tool(query: str, max_results: int = 3) -> tuple[str, list[Document]]:
    """
    Search the web for current or real time information.
    Adjust max_results (default 3) to control how many results are returned.
    """

    logger.info("Tool called: web search")
    response = tavily_client.search(query=query, max_results=max_results)

    docs = [
        Document(
            page_content=r.get("content"),
            metadata={"source": r.get("url"), "title": r.get("title", "")},
        )
        for r in response.get("results")
    ]

    content = "\n\n**Web Search Results**" + "\n\n".join(d.page_content for d in docs)

    return content, docs


# tool list
tools = [vector_store_search_tool, web_search_tool]

# bind tools with agent llm
agent_llm_with_tools = agent_llm.bind_tools(tools=tools)

# create tool node
tool_node = ToolNode(tools=tools)

# agent prompt: This is tutorial specific prompt and not generalized, it mentioned specific doc title, chapter names etc.
AGENT_SYSTEM_PROMPT = (
    "You are a retrieval agent with access to two tools:\n\n"
    "1. vector_store_search — use this for questions that can be answered from the internal document: "
    "a technical report titled 'Will EVs Dampen the Oil Price Shock?' covering EV adoption trajectories, "
    "oil demand displacement scenarios, fleet turnover dynamics, battery cost trends, OPEC+ supply behavior, "
    "and energy price volatility projections through 2050. "
    "Use this tool whenever the query references the report, its findings, its projections, or any topic "
    "that would plausibly appear in a domain-specific EV/oil-market research document. "
    "You may increase k beyond the default if broader coverage of the document is needed.\n\n"
    "2. web_search — use this for current or real-time information not covered by the document, such as "
    "recent market data, news, or statistics from 2024 onward. "
    "Always rephrase the query into a concise, keyword-optimized web search string before calling this tool.\n\n"
    "You may call one tool, both tools, or no tool depending on what the query requires. "
    "When both document knowledge and current data are relevant, call both tools."
)


# agent node: decides which tool to call based on the query
def agent_node(state: State) -> State:
    logger.info("Agent deciding which tool to call.")
    messages = state.get("messages")

    # DEBUG: checking messages
    for idx, msg in enumerate(messages):
        tool_call_info = (
            f" | Tool calls: {msg.tool_calls}" if hasattr(msg, "tool_calls") else ""
        )
        logger.debug(
            f"[{idx}] {msg.__class__.__name__}: {str(msg.content)[:80]}...{tool_call_info}"
        )

    if not messages:
        messages = [
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            HumanMessage(content=state.get("query")),
        ]
    response = agent_llm_with_tools.invoke(input=messages)
    return {"messages": [*messages, response]}


# collect tool output node: extract ToolMessages into state fields
def collect_tool_output_node(state: State) -> State:
    logger.info("Collecting tool output.")

    tool_messages = [m for m in state.get("messages") if isinstance(m, ToolMessage)]
    all_docs, context_parts = [], []

    for msg in tool_messages:
        context_parts.append(msg.content)
        all_docs.extend(msg.artifact)

    return {"context": "\n\n".join(context_parts), "retrieved_docs": all_docs}


# route after agent decision
def route_after_agent_decision(state: State) -> RouteVerdict:
    return (
        RouteVerdict.TOOLS_REQUIRED
        if state.get("messages")[-1].tool_calls
        else RouteVerdict.TOOLS_COMPLETE
    )


# build graph
graph = StateGraph(state_schema=State)

# add nodes
graph.add_node(Node.DECIDE_RETRIEVAL, decide_retrieval_node)
graph.add_node(Node.AGENT, agent_node)
graph.add_node(Node.TOOL, tool_node)
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
graph.add_edge(Node.TOOL, Node.AGENT)
graph.add_edge(Node.TOOL_OUTPUT, Node.GENERATE_RESPONSE)

# compile graph
app = graph.compile()

if __name__ == "__main__":
    ingest_documents()

    # Generates and writes the graph image directly to your project root / graphs
    png_data = app.get_graph().draw_mermaid_png()
    Path("graphs/agentic_rag_2.png").write_bytes(png_data)
    logger.info("Graph saved to root directory")

    result: State = app.invoke(
        {
            "query": "Explain the difference between kinetic energy and potential energy.",
            "messages": [],
            "context": "",
            "retrieved_docs": [],
        }
    )

    logger.info(f"Retrieval needed: {result.get('needs_retrieval')}")
    if result.get("retrieved_docs"):
        logger.info(f"Length of retrieved docs: {len(result.get('retrieved_docs'))}")
    else:
        logger.info("No docs retrieved.")
    logger.info(f"Query: {result.get('query')}")
    logger.info(f"Response: {result.get('response').content[0]['text']}")
