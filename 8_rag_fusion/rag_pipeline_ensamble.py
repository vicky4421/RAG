from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_fusion import RAGFusion
from rich.console import Console

load_dotenv()
console = Console()

# colors
primary_col = "yellow"
secondary_col = "cyan"

llm = init_chat_model("google_genai:gemini-3.1-flash-lite")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

# Load PDF
loader = PyPDFLoader(file_path="./notebooklm_rag.pdf")
pages = loader.load()
console.print(f"loaded {len(pages)} page(s) from PDF.", style=primary_col)

# Split documents
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents=pages)
console.print(f"Split into {len(chunks)} chunk(s).", style=primary_col)

# Embedding and vector store
vector_store = Chroma.from_documents(
    documents=chunks, embedding=embeddings, collection_name="notebook_rag_ensemble"
)
console.print("Vector store created!", style=primary_col)

# Create two retrivers, one is vanilla and second is MMR
similarity_retriver = vector_store.as_retriever(
    search_type="similarity", search_kwargs={"k": 3}
)

mmr_retriever = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 3, "lambda_mult": 0.5}
)

# RAG Fustion with ensemble retriever
rag_fusion = RAGFusion.from_retrievers(
    base_retrievers=[similarity_retriver, mmr_retriever], weights=[0.5, 0.5], k=3
)

query = "How does NotebookLM retrieve relevant information from uploaded documents?"

fused_docs = rag_fusion.invoke(query=query)

console.print(f"Retrieved {len(fused_docs)} fused document(s).", style=primary_col)

for i, doc in enumerate(fused_docs, start=1):
    console.print(f"{i}: {doc.page_content}", style=secondary_col)

# Augmentation
context = "\n\n".join([doc.page_content for doc in fused_docs])
console.print(f"\ncontext: {context}", style=primary_col)

# Generation
prompt = ChatPromptTemplate.from_template(
    template="""
    You are a helpful assistant. Use ONLY the context provided below to answer the question.
Be clear, concise, and accurate in your response.
If the answer is not present in the context, say "I don't know" - do not make up an answer.

Context:
{context}

Question: {question}

Answer:
"""
)

chain = prompt | llm

response = chain.invoke({"context": context, "question": query})

console.print(response.content, style=primary_col)

# loaded 3 page(s) from PDF.
# Split into 19 chunk(s).
# Vector store created!
# Retrieved 3 fused document(s).
# 1: model can reference the specific chunks it used to generate an answer.
# 3. How NotebookLM Processes Documents
# 2: vectors exist in the same semantic space, making cosine similarity a reliable measure of relevance.
# The system then performs a nearest-neighbor search against the vector index to retrieve the top-k most
# relevant document chunks. In practice, NotebookLM likely uses a combination of dense retrieval (vector
# similarity) and sparse retrieval such as keyword-based BM25 matching, through a technique known as hybrid
# 3: rather than simple keyword matching. The vectors are stored in a vector index that supports efficient
# nearest-neighbor search, enabling fast retrieval even across very large document collections.
# 4. Query Handling and Retrieval
# When a user submits a query in NotebookLM, the system converts the query into an embedding using the
# same model that was used to embed the document chunks. This ensures that the query and the document
# model can reference the specific chunks it used to generate an answer.
# 3. How NotebookLM Processes Documents

# vectors exist in the same semantic space, making cosine similarity a reliable measure of relevance.
# The system then performs a nearest-neighbor search against the vector index to retrieve the top-k most
# relevant document chunks. In practice, NotebookLM likely uses a combination of dense retrieval (vector
# similarity) and sparse retrieval such as keyword-based BM25 matching, through a technique known as hybrid

# rather than simple keyword matching. The vectors are stored in a vector index that supports efficient
# nearest-neighbor search, enabling fast retrieval even across very large document collections.
# 4. Query Handling and Retrieval
# When a user submits a query in NotebookLM, the system converts the query into an embedding using the
# same model that was used to embed the document chunks. This ensures that the query and the document
# Direct use of automatic function calling (AFC) in Models.generate_content is not recommended. Instead, we recommend to use AFC in Chat.send_message. Similarly, direct use of AFC in Models.generate_content_stream is not recommended. Instead, we recommend to use AFC in Chat.send_message_stream.
# [
#     {
#         'type': 'text',
#         'text': "NotebookLM retrieves information by converting a user's query into an embedding using the same model
# used for document chunks. It then performs a nearest-neighbor search against a vector index to retrieve the top-k most
# relevant document chunks. This process likely utilizes a combination of dense retrieval (vector similarity) and sparse
# retrieval (such as keyword-based BM25 matching), known as hybrid retrieval.",
#         'extras': {
#             'signature':
# 'EnEKbwERTTIPVYAlZggPGsMk7r/tPtM4Ze0s/YGbYWH50fSitDfZCRgibdoHgg7WdYa6cwtVFgIO8C/u1Zjn6O82AEIjySxF9xOfEqQQ223XIuFT3v5EKAE
# wqauKRuXiINJOZvNw30sECFg6SogGlP5HLQ=='
#         }
#     }
# ]
