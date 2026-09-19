from pathlib import Path
from typing import TypedDict

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from ch_11_self_rag.nodes import Node, RouteVerdict
from utils.logger import get_logger
from utils.utils import get_chroma_vector_store, get_llm, load_hf_embedding_model

logger = get_logger()

logger.info("Self RAG step_1 execution started.")

llm = get_llm()

embedding = load_hf_embedding_model(
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
)

vector_store = get_chroma_vector_store(
    collection_name="self_rag_demo",
    embedding_func=embedding,
    db_directory_name="chroma_db",
)

retriever = vector_store.as_retriever(search_kwargs={"k": 4})


def ingest_documents():
    # Avoid duplicate indexing if already populated
    count = vector_store._collection.count()
    if count > 0:
        logger.info(f"Collection already contains {count} chunks. Skipping ingestion.")
        return

    # load documents
    logger.info("Loading docs.")

    # Points directly to D:\AI\RAG\11_self_rag\documents
    DOCS_DIR = Path(__file__).resolve().parent / "documents"

    docs = (
        PyPDFLoader(str(DOCS_DIR / "Company_Policies.pdf")).load()
        + PyPDFLoader(str(DOCS_DIR / "Company_Profile.pdf")).load()
        + PyPDFLoader(str(DOCS_DIR / "Product_and_Pricing.pdf")).load()
    )

    # split docs
    logger.info("Splitting docs")
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=150
    ).split_documents(documents=docs)

    # add documents to vector
    vector_store.add_documents(documents=chunks)


# define graph state
class State(TypedDict):
    question: str
    need_retrieval: bool
    docs: list[Document]
    answer: str


# retrieve decision schema
class RetrieveDecision(BaseModel):
    """
    Schema defined to get structured output from llm to decide whether for given question retrieval is needed or not.
    """

    should_retrieve: bool = Field(
        description="True if external documents are needed to answer reliebly, else False"
    )


decide_retrieval_prompt = ChatPromptTemplate.from_messages(
    messages=[
        (
            "system",
            """You decide whether the retrieval needed.\n
        Return json that matches this schema: \n
        {{should_retrieve: boolean}}\n\n
        Guidelines:\n
        - should retrieve = True if answering requires specific facts, citations or info not likely in the mode.\n
        - should retrieve = False for general explanations, definations or reasoning that doesn't need resources.\n
        - if unsure, choose True""",
        ),
        ("human", "Question: {question}"),
    ]
)

# NOTE: no '.content' for structured output
should_retrieve_llm = llm.with_structured_output(RetrieveDecision)


def decide_retrieval_node(state: State) -> State:
    logger.info(msg="Deciding retrieval.")
    decision: RetrieveDecision = should_retrieve_llm.invoke(
        decide_retrieval_prompt.format_messages(question=state["question"])
    )
    return {"need_retrieval": decision.should_retrieve}


# direct generation prompt
direct_generation_prompt = ChatPromptTemplate.from_messages(
    messages=[
        (
            "system",
            """
            Answer the question using only your general knowledge.\n
            Do not assume access to external documents.\n
            If you're unsure or the answer requires specific sources, say: \n
            I don't know based on my general knowledge.
            """,
        ),
        ("human", "{question}"),
    ]
)


def generate_direct_node(state: State) -> State:
    logger.info(msg="Generating response without retrieval.")
    out = llm.invoke(
        direct_generation_prompt.format_messages(question=state["question"])
    )
    return {"answer": out.content}


def retrieve_node(state: State) -> State:
    logger.info(msg="Retrieving documents")
    return {"docs": retriever.invoke(state["question"])}


# routing condition
def route_after_decide(state: State) -> RouteVerdict:
    if state["need_retrieval"]:
        return RouteVerdict.RETRIEVE
    return RouteVerdict.GENERATE_DIRECT


# define graph
graph = StateGraph(state_schema=State)

# add nodes
graph.add_node(Node.DECIDE_RETRIEVAL, decide_retrieval_node)
graph.add_node(Node.GENERATE_DIRECT, generate_direct_node)
graph.add_node(Node.RETRIEVE, retrieve_node)

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
graph.add_edge(Node.RETRIEVE, END)

app = graph.compile()


if __name__ == "__main__":
    ingest_documents()

    # Generates and writes the graph image directly to your project root
    png_data = app.get_graph().draw_mermaid_png()
    Path("graphs/self_rag_1.png").write_bytes(png_data)
    print("Graph saved to root directory")

    result: State = app.invoke({"question": "Who is the CEO of Nexa AI"})
    print(f"Need retrieval? : {result['need_retrieval']}")

    if "answer" in result:
        print(result["answer"][0]["text"])
