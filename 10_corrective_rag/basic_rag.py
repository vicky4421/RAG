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
from tqdm import tqdm

load_dotenv()
console = Console()
logger = get_logger()
BATCH_SIZE = 50

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
# comment out after first run
# with console.status(status="Started loading docs...", spinner="clock"):
#     docs = (
#         # PyPDFLoader(file_path="./docs/book1.pdf").load()
#         # + PyPDFLoader(file_path="./docs/book2.pdf").load()
#         PyPDFLoader(file_path="./docs/book3.pdf").load()
#     )
# logger.info(msg=f"{len(docs)} documents loaded.")

# Step 2: Chunking
# comment out after first run
# chunks = RecursiveCharacterTextSplitter(
#     chunk_size=900, chunk_overlap=150
# ).split_documents(documents=docs)

# # Clean text to avoid UnicodeEncodeError (surrogates from PDF extraction)
# for d in chunks:
#     d.page_content = d.page_content.encode(encoding="utf-8", errors="ignore").decode(
#         encoding="utf-8", errors="ignore"
#     )
# logger.info(msg=f"{len(chunks)} chunks created.")

# Step 3: Vector Store


vector_store = Chroma(
    collection_name="corrective_rag_demo",
    embedding_function=embeddings,
    persist_directory=str(chroma_path),
)

# logger.info(msg=f"Embedding {len(chunks)} chunk(s)")

# for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding"):
#     batch = chunks[i : i + BATCH_SIZE]
#     vector_store.add_documents(documents=batch)
#     logger.info(f"Embedding successful for chunk(s) {i}")

# logger.info(msg="Embedding successful, Vector store created.")

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

# Output
# response: [{'type': 'text', 'text': 'A transformer is an estimator that can transform a
#                              dataset using the transform() method, which generally relies on learned parameters.',
#                              'extras': {'signature':
#                              'EnEKbwERTTIPGE6/5FnQp0Xx+VZoXVjjZyHJNMCosbu2XYEADKhzgynHKVTx3OayS3BPpdYIVnXgMNwKaj+Fwg0V8d
#                              plxvIUpi7zqo7JTmxPr1onk81lmDCjFngPVRsk1vmtfwwNHo4T6fhcOExCoZ3sDg=='}}]
# Retrieved docs:

# All transformers also have a convenience method called fit_transform()
# Prepare the Data for Machine Learning Algorithms | 61
# Download from finelybook www.finelybook.com
# ****************************************************************************************************
# performed by the fit() method, and it takes only a dataset as a parameter (or
# two for supervised learning algorithms; the second dataset contains the
# labels). Any other parameter needed to guide the estimation process is con‐
# sidered a hyperparameter (such as an imputer’s strategy), and it must be set
# as an instance variable (generally via a constructor parameter).
# — Transformers. Some estimators (such as an imputer ) can also transform a
# dataset; these are called transformers. Once again, the API is quite simple: the
# transformation is performed by the transform() method with the dataset to
# transform as a parameter. It returns the transformed dataset. This transforma‐
# tion generally relies on the learned parameters, as is the case for an imputer.
# All transformers also have a convenience method called fit_transform()
# Prepare the Data for Machine Learning Algorithms | 61
# ****************************************************************************************************
# 1. How would you define Machine Learning?
# 2. Can you name four types of problems where it shines?
# 3. What is a labeled training set?
# 4. What are the two most common supervised tasks?
# 5. Can you name four common unsupervised tasks?
# 6. What type of Machine Learning algorithm would you use to allow a robot to
# walk in various unknown terrains?
# 7. What type of algorithm would you use to segment your customers into multiple
# groups?
# 8. Would you frame the problem of spam detection as a supervised learning prob‐
# lem or an unsupervised learning problem?
# 9. What is an online learning system?
# 10.
# What is out-of-core learning?
# 11. What type of learning algorithm relies on a similarity measure to make predic‐
# tions?
# 12. What is the difference between a model parameter and a learning algorithm’s
# hyperparameter?
# 13. What do model-based learning algorithms search for? Wha
# t is the most common
# ****************************************************************************************************
# tf.int64, 265
# tf.In
# teractiveSession, 233
# TF .Learn, 264
# tf.log(), 427, 430, 446, 450
# tf.matmul(), 236-237, 246, 265, 384, 417,
# 420, 425, 427-428
# tf.matrix_inverse(), 236
# tf.maximum(), 246, 248-251, 281
# tf.multinomial(), 446, 450
# tf.name_scope(), 245, 248-249, 265,
# 267-268, 419-420
# tf.nn.conv2d(), 360-361
# tf.nn.dynamic_rnn(), 386-387, 390, 392,
# 395, 397-399, 409-410, 491-492
# tf.nn.elu(), 281, 416-417, 430, 446, 450
# tf.nn.embedding_lookup(), 406
# tf.nn.in_top_k(), 268, 391
# tf.nn.max_pool(), 364-365
# tf.nn.relu(), 265, 392-393, 395, 463
# tf.nn.sigmoid_cross_entropy_with_logits(),
# 428, 431, 449-450
# tf.nn.sparse_soft‐
# max_cross_entropy_with_logits(),
# 267-268, 390
# tf.one_hot(), 466
# tf.PaddingFIFOQueue, 334
# tf.placeholder(), 239-240, 482
# tf.placeholder_with_default(), 425
# tf.RandomShuffleQueue, 333, 337-338,
# 340-341
# tf.random_normal(), 246, 384, 425, 430
# tf.random_uniform(), 237, 241, 406, 482
