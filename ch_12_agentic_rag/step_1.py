from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel

from ch_12_agentic_rag.nodes import Node, RouteVerdict
from ch_12_agentic_rag.step_0 import ingest_documents, llm, retrieve_node
from utils.logger import get_logger

logger = get_logger()

logger.info("Agentic RAG step_1 execution started.")


# state
class State(MessagesState):
    query: str
    response: str
    context: str
    needs_retrieval: bool
    retrieved_docs: list[Document]


# schema for routing decision: structured output from llm
class RetrieveDecision(BaseModel):
    needs_retrieval: bool


# node decide retrieval
def decide_retrieval_node(state: State) -> State:
    logger.info("Deciding whether to retrieve documents or not.")
    prompt_template = ChatPromptTemplate(
        messages=[
            (
                "system",
                "Classify whether the following question requires retrieving information from a specialized document, or can be answered from your own general knowledge.",
            ),
            ("human", "{query}"),
        ]
    )
    decision: RetrieveDecision = (
        prompt_template | llm.with_structured_output(schema=RetrieveDecision)
    ).invoke({"query": state.get("query")})

    return {"needs_retrieval": decision.needs_retrieval}


# node generate response from context or direct
def generate_node(state: State) -> State:
    logger.info("Generating response.")

    query = state.get("query")
    context = state.get("context", "")

    # if retrieval needed and retrieved docs are added to context
    if context:
        prompt = ChatPromptTemplate.from_messages(
            messages=[
                (
                    "system",
                    "Answer the question using only the context below. \n\nContext:\n{context}",
                ),
                ("human", "{query}"),
            ]
        )
        response = (prompt | llm).invoke({"context": context, "query": query})

    # if retrieval is not needed
    else:
        prompt = ChatPromptTemplate.from_messages(
            messages=[
                (
                    "system",
                    "Answer the following question using your general knowledge",
                ),
                ("human", "{query}"),
            ]
        )
        response = (prompt | llm).invoke({"query": query})

    return {"response": response}


# route after retrieval decide
def route_after_retrieval_decide(state: State) -> RouteVerdict:
    return (
        RouteVerdict.NEED_RETRIEVAL
        if state.get("needs_retrieval")
        else RouteVerdict.GENERATE_DIRECT
    )


# define graph
graph = StateGraph(state_schema=State)

# add nodes
graph.add_node(Node.RETRIEVE, retrieve_node)
graph.add_node(Node.GENERATE_RESPONSE, generate_node)
graph.add_node(Node.DECIDE_RETRIEVAL, decide_retrieval_node)

# add edges
graph.add_edge(START, Node.DECIDE_RETRIEVAL)
graph.add_conditional_edges(
    source=Node.DECIDE_RETRIEVAL,
    path=route_after_retrieval_decide,
    path_map={
        RouteVerdict.NEED_RETRIEVAL: Node.RETRIEVE,
        RouteVerdict.GENERATE_DIRECT: Node.GENERATE_RESPONSE,
    },
)
graph.add_edge(Node.GENERATE_RESPONSE, END)
graph.add_edge(Node.RETRIEVE, Node.GENERATE_RESPONSE)

# compile graph
app = graph.compile()

if __name__ == "__main__":
    ingest_documents()

    # Generates and writes the graph image directly to your project root / graphs
    png_data = app.get_graph().draw_mermaid_png()
    Path("graphs/agentic_rag_1.png").write_bytes(png_data)
    logger.info("Graph saved to root directory")

    result: State = app.invoke(
        {
            "query": "What is the capital of india?",
            "messages": [],
        }
    )

    logger.info("Response")
    logger.info(result.get("response"))
