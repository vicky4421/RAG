from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_chroma.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from logger import get_logger
from rich.console import Console

load_dotenv()
console = Console()
logger = get_logger()

# colors
primary_col = "yellow"
secondary_col = "cyan"

# cwd
chroma_path = Path(__file__).resolve().parent / "chroma_db"

# directory for embeddinggemma model
shared_embedding_cache = r"c:\ai_models_cache\huggingface"

logger.info(msg="Pipeline initiated.")

llm = init_chat_model("google_genai:gemini-3.1-flash-lite")

embeddings = HuggingFaceEmbeddings(
    model_name="google/embeddinggemma-300m",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    cache_folder=shared_embedding_cache,
)

# Step 1: Load Documents
with console.status(status="Started loading docs...", spinner="clock"):
    docs = (
        PyPDFLoader(file_path="./docs/book1.pdf").load()
        + PyPDFLoader(file_path="./docs/book2.pdf").load()
        + PyPDFLoader(file_path="./docs/book3.pdf").load()
    )
logger.info(msg=f"{len(docs)} documents loaded.")

# Step 2: Chunking
chunks = RecursiveCharacterTextSplitter(
    chunk_size=900, chunk_overlap=150
).split_documents(documents=docs)

# Clean text to avoid UnicodeEncodeError (surrogates from PDF extraction)
for d in chunks:
    d.page_content = d.page_content.encode(encoding="utf-8", errors="ignore").decode(
        encoding="utf-8", errors="ignore"
    )
logger.info(msg=f"{len(chunks)} chunks created.")

# Step 3: Vector Store

if chroma_path.exists():
    logger.info(msg="Loading existing Chroma index from disk...")
    vector_store = Chroma(
        collection_name="corrective_rag_demo",
        embedding_function=embeddings,
        persist_directory=str(chroma_path),
    )
    logger.info(msg="Loaded from disk instantly!")
else:
    logger.info("Embedding documents starts for 100 chunks")
    vector_store = Chroma.from_documents(
        documents=chunks[:100],
        embedding=embeddings,
        collection_name="corrective_rag_demo",
        persist_directory=str(chroma_path),
    )
    logger.info(msg="Embedding successful, Vector store created.")

# Step 4: Retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
logger.info("Retriever created")


# define state schema
class State(TypedDict):
    question: str
    docs: list[Document]
    answer: str


# retrieval
def retrieve(state: State) -> State:
    q = state["question"]
    return {"docs": retriever.invoke(input=q)}


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer only from context. If not in context say you don't know"),
        ("human", "Question: {question}\n\nContext: {context}"),
    ]
)


# Step 5: Generation
def generate(state: State) -> State:
    context = "\n\n".join(d.page_content for d in state["docs"])
    out = (prompt | llm).invoke({"question": state["question"], "context": context})
    return {"answer": out.content}


# initiate graph
graph = StateGraph(state_schema=State)

# add nodes
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)

# add edges
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

# compile graph
app = graph.compile()

res = app.invoke({"question": "What is a transformer in deep learning?"})
logger.info(msg=f"Response: {res['answer']}")

console.print("Retrieved docs: \n", style=primary_col)
console.print(res["docs"][0].page_content, style=secondary_col)
console.print("*" * 100, style="blue")
console.print(res["docs"][1].page_content, style=secondary_col)
console.print("*" * 100, style="blue")
console.print(res["docs"][2].page_content, style=secondary_col)
console.print("*" * 100, style="blue")
console.print(res["docs"][3].page_content, style=secondary_col)
