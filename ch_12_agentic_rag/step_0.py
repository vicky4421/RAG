from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, MessagesState, StateGraph

from ch_12_agentic_rag.nodes import Node
from utils.logger import get_logger
from utils.utils import get_chroma_vector_store, get_llm, load_hf_embedding_model

logger = get_logger()

logger.info("Agentic RAG step_1 execution started.")

llm = get_llm()

embedding = load_hf_embedding_model(
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
)

vector_store = get_chroma_vector_store(
    collection_name="agentic_rag_demo",
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

    docs = PyPDFLoader(str(DOCS_DIR / "evs_oil_price_shock.pdf")).load()

    # split docs
    logger.info("Splitting docs")
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=150
    ).split_documents(documents=docs)

    # add documents to vector
    vector_store.add_documents(documents=chunks)


# define state using MessageState and not typeddict, coz messagestate keeps record / store all messages and we don't have to explicitly have to manage messages / best for agents
class State(MessagesState):
    query: str
    retrieved_docs: list[Document]
    context: str
    response: str


# retrieve node
def retrieve_node(state: State) -> State:
    docs = retriever.invoke(state.get("query"))
    context = "\n\n".join(doc.page_content for doc in docs)
    return {"retrieved_docs": docs, "context": context}


# generate node
def generate_node(state: State) -> State:
    query = state.get("query")
    context = state.get("context", "")
    prompt_template = ChatPromptTemplate.from_messages(
        messages=[
            (
                "system",
                "Answer the question using only the context below. \n\nContext: \n{context}",
            ),
            ("human", "{query}"),
        ]
    )
    response = (prompt_template | llm).invoke({"context": context, "query": query})
    return {"response": response.content}


# define graph
graph = StateGraph(state_schema=State)

# add node
graph.add_node(Node.RETRIEVE, retrieve_node)
graph.add_node(Node.GENERATE_RESPONSE, generate_node)

# add edges
graph.add_edge(START, Node.RETRIEVE)
graph.add_edge(Node.RETRIEVE, Node.GENERATE_RESPONSE)
graph.add_edge(Node.GENERATE_RESPONSE, END)

# compile graph
app = graph.compile()

if __name__ == "__main__":
    ingest_documents()

    # Generates and writes the graph image directly to your project root / graphs
    png_data = app.get_graph().draw_mermaid_png()
    Path("graphs/agentic_rag_0.png").write_bytes(png_data)
    logger.info("Graph saved to root directory")

    result: State = app.invoke(
        {"query": "How will EVs impact oil demand in the next decade?", "messages": []}
    )

    logger.info("Response")
    logger.info(result.get("response")[0]["text"])
