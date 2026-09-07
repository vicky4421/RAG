import warnings

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_fusion import RAGFusion
from rich.console import Console

# Silence the Google AFC notice
warnings.filterwarnings("ignore", category=UserWarning, module="google")

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
    documents=chunks, embedding=embeddings, collection_name="notebook_rag"
)
console.print("Vector store created!", style=primary_col)

# Create one retrivers
similarity_retriver = vector_store.as_retriever(
    search_type="similarity", search_kwargs={"k": 3}
)

# RAG Fustion with ensemble retriever
rag_fusion = RAGFusion.from_llm(llm=llm, retriever=similarity_retriver)

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

# Retrieved 5 fused document(s).
# 1: How NotebookLM is Performing RAG Under the Hood
# 1. Introduction to NotebookLM
# NotebookLM is an AI-powered research and note-taking tool developed by Google. It is designed to help
# users understand complex documents, generate insights, and answer questions based on content that users
# upload directly. Unlike general-purpose AI assistants, NotebookLM grounds all of its responses in the source
# 2: vectors exist in the same semantic space, making cosine similarity a reliable measure of relevance.
# The system then performs a nearest-neighbor search against the vector index to retrieve the top-k most
# relevant document chunks. In practice, NotebookLM likely uses a combination of dense retrieval (vector
# similarity) and sparse retrieval such as keyword-based BM25 matching, through a technique known as hybrid
# 3: generation, it enables users to extract reliable insights from their own source materials without the risk of
# hallucination that plagues general-purpose AI assistants.
# Understanding the RAG pipeline that powers NotebookLM -from document chunking and embedding to
# retrieval and grounded generation -provides a solid foundation for building similar systems. The principles
# covered in this document are directly applicable to custom RAG implementations using frameworks such as
# 4: answering questions.
# At its core, NotebookLM relies on a sophisticated implementation of Retrieval-Augmented Generation (RAG)
# to ensure that responses are accurate, grounded, and traceable back to the original source material.
# Understanding how this works under the hood provides valuable insight into both the tool's capabilities and its
# limitations.
# 2. What is Retrieval-Augmented Generation (RAG)?
# 5: rather than simple keyword matching. The vectors are stored in a vector index that supports efficient
# nearest-neighbor search, enabling fast retrieval even across very large document collections.
# 4. Query Handling and Retrieval
# When a user submits a query in NotebookLM, the system converts the query into an embedding using the
# same model that was used to embed the document chunks. This ensures that the query and the document

# context: How NotebookLM is Performing RAG Under the Hood
# 1. Introduction to NotebookLM
# NotebookLM is an AI-powered research and note-taking tool developed by Google. It is designed to help
# users understand complex documents, generate insights, and answer questions based on content that users
# upload directly. Unlike general-purpose AI assistants, NotebookLM grounds all of its responses in the source

# vectors exist in the same semantic space, making cosine similarity a reliable measure of relevance.
# The system then performs a nearest-neighbor search against the vector index to retrieve the top-k most
# relevant document chunks. In practice, NotebookLM likely uses a combination of dense retrieval (vector
# similarity) and sparse retrieval such as keyword-based BM25 matching, through a technique known as hybrid

# generation, it enables users to extract reliable insights from their own source materials without the risk of
# hallucination that plagues general-purpose AI assistants.
# Understanding the RAG pipeline that powers NotebookLM -from document chunking and embedding to
# retrieval and grounded generation -provides a solid foundation for building similar systems. The principles
# covered in this document are directly applicable to custom RAG implementations using frameworks such as

# answering questions.
# At its core, NotebookLM relies on a sophisticated implementation of Retrieval-Augmented Generation (RAG)
# to ensure that responses are accurate, grounded, and traceable back to the original source material.
# Understanding how this works under the hood provides valuable insight into both the tool's capabilities and its
# limitations.
# 2. What is Retrieval-Augmented Generation (RAG)?

# rather than simple keyword matching. The vectors are stored in a vector index that supports efficient
# nearest-neighbor search, enabling fast retrieval even across very large document collections.
# 4. Query Handling and Retrieval
# When a user submits a query in NotebookLM, the system converts the query into an embedding using the
# same model that was used to embed the document chunks. This ensures that the query and the document
# [
#     {
#         'type': 'text',
#         'text': 'NotebookLM retrieves relevant information by converting user queries into embeddings using the same
# model used for document chunks. It then performs a nearest-neighbor search against a vector index to retrieve the top-k
# most relevant chunks. The system likely uses a hybrid approach, combining dense retrieval (vector similarity) and sparse
# retrieval (such as keyword-based BM25 matching).',
#         'extras': {
#             'signature':
# 'EnEKbwERTTIP6PLwZv/ygMSTQOaC3eTbvSsnITvaC3n1k+woo+Y2ZPpHinW0pLUPwl5dz5xX9c0UnQj3U/rBZw0YqzdYrgqJAvizS4jxgDSmHAspwjr7vZx
# kE0gle7fovinYZylDXDmp4jE57hMd8yYcfA=='
#         }
#     }
# ]
