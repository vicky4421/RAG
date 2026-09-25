# define state
import logging
import operator
from pathlib import Path
from typing import Annotated

from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel

from ch_12_agentic_rag.nodes import Node, RouteVerdict
from ch_12_agentic_rag.step_0 import ingest_documents, llm
from ch_12_agentic_rag.step_1 import (
    decide_retrieval_node,
    route_after_retrieval_decision,
)
from ch_12_agentic_rag.step_2 import agent_llm_with_tools, tool_node
from ch_12_agentic_rag.step_4 import (
    RelevanceEvaluation,
)
from utils.logger import get_logger

# silent gemini warnings
logging.getLogger(name="google_genai.models").setLevel(level=logging.ERROR)

logger = get_logger()

logger.info("Agentic RAG step_5 execution started.")


class State(MessagesState):
    query: str
    response: str
    retrieved_docs: Annotated[list[Document], operator.add]
    relevant_docs: list[Document]
    context: str
    needs_retrieval: bool
    tool_call_count: int
    max_tool_call: int
    is_relevant: bool
    rewritten_query: str
    rewrite_count: int
    needs_decomposition: bool


# schema for decomposition check
class DecompositionDecision(BaseModel):
    needs_decomposition: bool
    sub_queries: list[str]


# check decomposition node
def decide_decomposition_node(state: State) -> State:
    logger.info("Checking if query needed decomposition")
    prompt = ChatPromptTemplate.from_messages(
        messages=[
            (
                "system",
                """
            You're a query analysis assistant. Determine if users query asks for multiple distinct pieces of information that each require separate retrieval.
            If yes, needs_decomposition should be True.
            If the question is simple or self contained, needs_decomposition should be False.
            """,
            ),
            ("human", "{query}"),
        ]
    )
    chain = prompt | llm.with_structured_output(DecompositionDecision)
    result: DecompositionDecision = chain.invoke({"query": state.get("query")})
    return {"needs_decomposition": result.needs_decomposition}


# decompose query node
def decompose_query_node(state: State) -> State:
    logger.info("Decomposing user query.")
    prompt = ChatPromptTemplate.from_messages(
        messages=[
            (
                "system",
                """
            You're a query decomposition assistant. Break the user's question into focused, self-contained sub queries - one per retrieval step.
            Each subquery should target a single distinct piece of information.
            """,
            ),
            ("human", "{query}"),
        ]
    )
    chain = prompt | llm.with_structured_output(DecompositionDecision)
    result: DecompositionDecision = chain.invoke({"query": state.get("query")})
    formatted = "\n".join(
        f"Step {i + 1}: {q}" for i, q in enumerate(result.sub_queries)
    )
    logger.info(f"Subqueries: {formatted}")
    return {"query": formatted}


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
    "When both document knowledge and current data are relevant, call both tools.\n\n"
    "When the query contains numbered steps (e.g. Step 1: ..., Step 2: ...), each step is a separate "
    "retrieval sub-task. Make sure every step receives at least one retrieval call — do not skip or "
    "overlook any step. For each step, choose whichever tool or combination of tools best fits the "
    "information need; the retrieval path for each step is yours to decide."
)


# agent node: optimized for rewritten query and updated AGENT_SYSTEM_PROMPT.
def agent_node(state: State) -> State:
    logger.info("Agent deciding which tool to call.")

    # use rewritten query if available to start a fresh retrieval cycle
    rewritten_query = state.get("rewritten_query")

    if rewritten_query:
        messages = [
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            HumanMessage(content=rewritten_query),
        ]
    else:
        messages = state.get("messages")
        if not messages:
            messages = [
                SystemMessage(content=AGENT_SYSTEM_PROMPT),
                HumanMessage(content=state.get("query")),
            ]

    response = agent_llm_with_tools.invoke(messages)
    return {"messages": [*messages, response]}


# same as previous but state is different, problem with relevant_docs
def increment_tool_call_count_node(state: State) -> State:
    logger.info("Incrementing tool call count.")
    count = state.get("tool_call_count", 0)
    return {"tool_call_count": count + 1}


# collect tool output node: extract ToolMessages into state fields, no context generation this time
# same as before but type of retrieved docs in state is changed in this step
def collect_tool_output_node(state: State) -> State:
    logger.info("Collecting tool output.")

    all_docs = []
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            break
        if isinstance(msg, ToolMessage):
            all_docs.extend(msg.artifact)
    all_docs.reverse()
    return {"retrieved_docs": all_docs}


# evaluation node to evaluate the retrieved docs are relevant
# same as before but state changed, change: type of relevant_docs
def evaluate_docs_node(state: State) -> State:
    logger.info("Evaluating docs.")

    docs = state.get("retrieved_docs") or []

    if not docs:
        return {"is_relevant": False, "relevant_docs": [], "context": ""}

    active_query = state.get("rewritten_query") or state.get("query")

    prompt = ChatPromptTemplate.from_messages(
        messages=[
            (
                "system",
                """
            You're a relevance evaluator. Given a user query and a single document passage.
            Determine whether the passage contains useful information to answer the query.
            Return is_relevant=True only if the passage directly addresses the query with specific facts or analysis.
            Return is_relevant=False if it is off topic, too vague, or doesn't help to answer the query.
            """,
            ),
            ("human", "Query:\n{query}\nDocument passage: \n{doc}"),
        ]
    )

    chain = prompt | llm.with_structured_output(schema=RelevanceEvaluation)

    relevant_docs = []

    for doc in docs:
        result: RelevanceEvaluation = chain.invoke(
            {"query": active_query, "doc": doc.page_content}
        )
        if result.is_relevant:
            relevant_docs.append(doc)

    is_relevant = len(relevant_docs) > 0
    context = "\n\n".join(d.page_content for d in relevant_docs)

    return {
        "is_relevant": is_relevant,
        "relevant_docs": relevant_docs,
        "context": context,
    }


# rewrite query node
# same as before but state changed, change: type of relevant_docs
def rewrite_query_node(state: State) -> State:
    logger.info("Rewriting query.")

    count = state.get("rewrite_count", 0)
    prompt = ChatPromptTemplate.from_messages(
        messages=[
            (
                "system",
                """
            You're a query rewritting assistant. A retrieval system searched for information to answer the user's query
            but none of the retrieved documents were relevant.
            Your task is to rewrite the query to be more specific, use different terminology, or focus on a norrower
            aspect that is more likely to match content in the knowledge source.
            Return only the rewritten query string.
            """,
            ),
            ("human", "Original query: \n{query}\n\nRewritten Query: "),
        ]
    )
    chain = prompt | llm
    response = chain.invoke({"query": state.get("query")})
    new_query = response.content
    logger.info(f"New query: {new_query}")
    return {"rewritten_query": new_query, "rewrite_count": count + 1}


# generate node with fallback when all rewrite attempts are exhausted with no relevant docs
# same as before but state changed, change: type of relevant_docs
def generate_node(state: State) -> State:
    logger.info("Generating response")

    if state.get("rewrite_count", 0) >= 3 and not state.get("is_relevant"):
        return {
            "response": (
                """
                I was unable to find relevant information to answer your query after multiple retrieval attempts.
                The knowledge source doesn't appear to contain content that addresses this query.
                """
            )
        }

    query = state.get("query")
    context = state.get("context") or ""

    if context:
        prompt = ChatPromptTemplate.from_messages(
            messages=[
                (
                    "system",
                    "Answer the question using only the context below.\n\nContext: \n{context}",
                ),
                ("human", "{query}"),
            ]
        )
        response = (prompt | llm).invoke({"context": context, "query": query})
    else:
        prompt = ChatPromptTemplate.from_messages(
            messages=[
                ("system", "Answer the following query using your general knowledge."),
                ("human", "{query}"),
            ]
        )
        response = (prompt | llm).invoke({"query": query})

    return {"response": response}


# same as previous but state is different, problem with context
def route_after_agent_decision(state: State) -> RouteVerdict:
    messages = state.get("messages", [])
    if messages and getattr(messages[-1], "tool_calls", None):
        return RouteVerdict.TOOLS_REQUIRED
    return RouteVerdict.TOOLS_COMPLETE


# same as previous but state is different, problem with context
def route_after_tool_call_limit_check(state: State) -> RouteVerdict:
    current = state.get("tool_call_count", 0)
    limit = state.get("max_tool_call_count", 3)
    return (
        RouteVerdict.TOOL_CALL_LIMIT_REACHED
        if current >= limit
        else RouteVerdict.TOOL_CALL_WITHIN_LIMIT
    )


# route after evaluation
def route_after_evaluation(state: State) -> RouteVerdict:
    return (
        RouteVerdict.DOCS_RELEVANT
        if state.get("is_relevant")
        else RouteVerdict.DOCS_NOT_RELEVANT
    )


# route after rewriting query
def route_after_rewrite_decision(state: State) -> RouteVerdict:
    return (
        RouteVerdict.REWRITE_EXHAUSTED
        if state.get("rewrite_count", 0) >= 3
        else RouteVerdict.REWRITE_NEEDED
    )


# route after decomposition
def route_after_decomposition_check(state: State) -> RouteVerdict:
    return (
        RouteVerdict.DECOMPOSITION_NEEDED
        if state.get("needs_decomposition")
        else RouteVerdict.DECOMPOSITION_NOT_NEEDED
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
graph.add_node(Node.GENERATE_RESPONSE, generate_node)

# add edges
graph.add_edge(START, Node.DECIDE_RETRIEVAL)
graph.add_conditional_edges(
    source=Node.DECIDE_RETRIEVAL,
    path=route_after_retrieval_decision,
    path_map={
        RouteVerdict.RETRIEVAL_NEEDED: Node.DECIDE_DECOMPOSITION,
        RouteVerdict.GENERATE_DIRECT: Node.GENERATE_RESPONSE,
    },
)
graph.add_edge(Node.GENERATE_RESPONSE, END)
graph.add_conditional_edges(
    source=Node.DECIDE_DECOMPOSITION,
    path=route_after_decomposition_check,
    path_map={
        RouteVerdict.DECOMPOSITION_NEEDED: Node.DECOMPOSE,
        RouteVerdict.DECOMPOSITION_NOT_NEEDED: Node.AGENT,
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
graph.add_edge(Node.TOOL_OUTPUT, Node.EVALUATE_DOCS)
graph.add_edge(Node.TOOL, Node.INCREMENT_TOOL_CALL)
graph.add_conditional_edges(
    source=Node.INCREMENT_TOOL_CALL,
    path=route_after_tool_call_limit_check,
    path_map={
        RouteVerdict.TOOL_CALL_LIMIT_REACHED: Node.TOOL_OUTPUT,
        RouteVerdict.TOOLS_REQUIRED: Node.AGENT,
    },
)
graph.add_conditional_edges(
    source=Node.EVALUATE_DOCS,
    path=route_after_evaluation,
    path_map={
        RouteVerdict.DOCS_RELEVANT: Node.GENERATE_RESPONSE,
        RouteVerdict.DOCS_NOT_RELEVANT: Node.REWRITE_QUERY,
    },
)
graph.add_conditional_edges(
    source=Node.REWRITE_QUERY,
    path=route_after_rewrite_decision,
    path_map={
        RouteVerdict.REWRITE_EXHAUSTED: Node.GENERATE_RESPONSE,
        RouteVerdict.REWRITE_NEEDED: Node.AGENT,
    },
)

# compile graph
app = graph.compile()

if __name__ == "__main__":
    ingest_documents()

    # Generates and writes the graph image directly to your project root / graphs
    png_data = app.get_graph().draw_mermaid_png()
    Path("graphs/agentic_rag_5.png").write_bytes(png_data)
    logger.info("Graph saved to root directory")

    result: State = app.invoke(
        {
            "query": "Impact of EVs on oil industry, and what is the current price of oil in USD for brent crude?",
            "messages": [],
            "context": "",
            "retrieved_docs": [],
            "max_tool_call_count": 5,
            "rewrite_count": 0,
        }
    )

    logger.info(f"Retrieval needed: {result.get('needs_retrieval')}")
    if result.get("retrieved_docs"):
        logger.info(f"Length of retrieved docs: {len(result.get('retrieved_docs'))}")
    else:
        logger.info("No docs retrieved.")
    logger.info(f"No. of tool calls: {result.get('tool_call_count')}")
    logger.info(f"Rewrite counts: {result.get('rewrite_count')}")
    logger.info(f"Query: {result.get('query')}")
    logger.info(f"Response:\n {result.get('response').content[0]['text']}")
