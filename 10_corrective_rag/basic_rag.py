from typing import TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from rich.console import Console

load_dotenv()
console = Console()

# colors
primary_col = "yellow"
secondary_col = "cyan"

llm = init_chat_model("google_genai:gemini-3.1-flash-lite")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

# Step 1: Load Documents
docs = (
    PyPDFLoader(file_path="./docs/book1.pdf").load()
    + PyPDFLoader(file_path="./docs/book2.pdf").load()
    + PyPDFLoader(file_path="./docs/book3.pdf").load()
)
console.print(f"\n{len(docs)} documents loaded.", style=primary_col)

# Step 2: Chunking
chunks = RecursiveCharacterTextSplitter(
    chunk_size=900, chunk_overlap=150
).split_documents(documents=docs)

# Clean text to avoid UnicodeEncodeError (surrogates from PDF extraction)
for d in chunks:
    d.page_content = d.page_content.encode(encoding="utf-8", errors="ignore").decode(
        encoding="utf-8", errors="ignore"
    )
console.print(f"{len(chunks)} created.", style=primary_col)

# Step 3: Vector Store
vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
console.print("Vector store created.", style=primary_col)

# Step 4: Retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
console.print("Retriever created.", style=primary_col)


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
console.print(res["answer"], style=secondary_col)

console.print("Retrieved docs: \n", style=primary_col)
console.print(res["docs"][0].page_content, style=secondary_col)
console.print("*" * 100, style="blue")
console.print(res["docs"][1].page_content, style=secondary_col)
console.print("*" * 100, style="blue")
console.print(res["docs"][2].page_content, style=secondary_col)
console.print("*" * 100, style="blue")
console.print(res["docs"][3].page_content, style=secondary_col)
