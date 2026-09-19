from pathlib import Path
from typing import TypedDict

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
from utils.logger import get_logger

logger = get_logger()

logger.info("Self RAG step_2 execution started.")


class State(TypedDict):
    question: str
    need_retrieval: bool
    docs: list[Document]
    relevant_docs: list[Document]
    answer: str


# relevance schema
class RelevanceDecision(BaseModel):
    is_relevant: bool = Field(
        description="True if document helps answer the question else False"
    )


is_relevant_prompt = ChatPromptTemplate.from_messages(
    messages=[
        (
            "system",
            """
                You're judging document relevance.\n
                Return JSON that matches this scheme.\n
                {{'is_relevant': boolean}}\n
                A document is relevant if it contains information useful for answering question.
            """,
        ),
        ("human", "Question: {question}\n\nDocument:\n{document}"),
    ]
)

relevance_llm = llm.with_structured_output(RelevanceDecision)


# is relevant node
def is_relevant_node(state: State) -> State:
    logger.info("Checking if documents are relevant.")
    relevant_docs: list[Document] = []
    for doc in state["docs"]:
        decision: RelevanceDecision = relevance_llm.invoke(
            is_relevant_prompt.format_messages(
                question=state["question"], document=doc.page_content
            )
        )
        if decision.is_relevant:
            relevant_docs.append(doc)

    return {"relevant_docs": relevant_docs}


# create graph
graph = StateGraph(state_schema=State)

# add nodes
graph.add_node(Node.DECIDE_RETRIEVAL, decide_retrieval_node)
graph.add_node(Node.GENERATE_DIRECT, generate_direct_node)
graph.add_node(Node.RETRIEVE, retrieve_node)
graph.add_node(Node.IS_RELEVANT, is_relevant_node)

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
graph.add_edge(Node.IS_RELEVANT, END)

# compile graph
app = graph.compile()

if __name__ == "__main__":
    # Generates and writes the graph image directly to your project root/graph
    png_data = app.get_graph().draw_mermaid_png()
    Path("graphs/self_rag_2.png").write_bytes(png_data)
    print("Graph saved to root/graph directory")

    result: State = app.invoke({"question": "Who is the CEO of NexaAI"})

    print(f"keys in result: {result.keys()}")

    print(f"Need retrieval? : {result['need_retrieval']}")

    if "answer" in result:
        print(result["answer"][0]["text"])
    else:
        print("Relevant Documents:\n")
        for i, doc in enumerate(result["relevant_docs"], start=1):
            print(f"{i}: {doc}")
