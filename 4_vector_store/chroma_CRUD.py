import shutil
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# get project root path
project_root = Path(__file__).parent.parent

load_dotenv()

collection_name = "demo"
persist_directory = project_root / "4_vector_store" / "chroma_db"

# overwrite previous db
if persist_directory.exists():
    shutil.rmtree(path=persist_directory)
    print("Overwriting old database!")
else:
    print("New database created!")

# create embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
)

# RUN FIRST TIME ONLY
# create vector store
vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=embeddings,
    persist_directory=str(persist_directory),
)

print("Vector store is ready!")


# helper function to print document
def print_docs(title, docs: list[Document]):
    """Print document objects in simple format"""
    print(f"title: {title}")
    for index, doc in enumerate(iterable=docs, start=1):
        print(f"{index}. id: {doc.id}")
        print(
            f"topic: {doc.metadata['topic']} \ndoc_number: {doc.metadata['doc_number']}"
        )
        print(f"content: {doc.page_content}")
    print()


# raw data (not Documents)
raw_data = [
    {
        "topic": "AI",
        "doc_number": 1,
        "text": "Artificial intelligence helps machines perform tasks that usually need human reasoning.",
    },
    {
        "topic": "AI",
        "doc_number": 2,
        "text": "AI systems can analyze patterns in data to support predictions and automation.",
    },
    {
        "topic": "AI",
        "doc_number": 3,
        "text": "Responsible AI development includes fairness, transparency, and safety checks.",
    },
    {
        "topic": "RAG",
        "doc_number": 4,
        "text": "RAG combines retrieval with generation so the model can answer using external knowledge.",
    },
    {
        "topic": "RAG",
        "doc_number": 5,
        "text": "A retriever in a RAG pipeline finds relevant chunks before the language model generates an answer.",
    },
    {
        "topic": "RAG",
        "doc_number": 6,
        "text": "Vector stores are important in RAG because they make semantic search over embedded documents possible.",
    },
    {
        "topic": "LLM",
        "doc_number": 7,
        "text": "LLMs generate text by predicting likely next tokens from patterns learned during training.",
    },
    {
        "topic": "LLM",
        "doc_number": 8,
        "text": "Prompt design can improve how clearly an LLM follows instructions and returns useful answers.",
    },
    {
        "topic": "Cricket",
        "doc_number": 9,
        "text": "Cricket teams score runs through batting partnerships, boundaries, and quick running between the wickets.",
    },
    {
        "topic": "Cricket",
        "doc_number": 10,
        "text": "A cricket bowler can pressure batters with pace, swing, spin, and accurate line and length.",
    },
]

print(f"Prepared {len(raw_data)} document examples.")

# conver raw data into Document objects using list comprehension
documents = [
    Document(
        id=str(uuid4()),
        page_content=item["text"],
        metadata={"topic": item["topic"], "doc_number": item["doc_number"]},
    )
    for item in raw_data
]
print("Document objects are ready!")

# RUN FIRST TIME ONLY
# insert documents in chromaDB
document_ids = vector_store.add_documents(documents=documents)
print("Documents added: \n")
for id in document_ids:
    print(id)

print(f"Total docs added: {len(document_ids)}")

# Read / Retrieve
raw_records = vector_store.get(include=["embeddings", "metadatas", "documents"])

print(raw_records.keys())

# print raw data from vector store
# print(vector_store.get())

# print embeddings, metadatas and documents
# print(raw_records)

print(f"\nTotal records in collection: {len(raw_records['ids'])}")
print("First 3 ids: \n")
for id in raw_records["ids"][:3]:
    print(id)

# get by ids
print_docs(
    title="Documents fetched with get_by_ids func: ",
    docs=vector_store.get_by_ids(raw_records["ids"][:3]),
)

# SIMILARITY SEARCH
query = "How does RAG help an llm answer questions using outside knowledge?"

search_result = vector_store.similarity_search(query=query, k=3)
print(f"Query: {query}")
print_docs(title="Similarity search results: \n", docs=search_result)

# Similarity search with score
# NOTE: In ChromaDB, the returned score represents distance by default, so a lower score means higher similarity
print(
    f"Similarity search with score \n {vector_store.similarity_search_with_score(query=query, k=3)}"
)

# Similarity search with score
# [(Document(id='5fcd47fb-6c72-4967-b141-24709e97e930', metadata={'topic': 'RAG', 'doc_number': 4}, page_content='RAG combines retrieval with generation so the model can answer using external knowledge.'), 0.2854572534561157), (Document(id='a6e44d6d-e9e5-4b33-b196-6c9a6b66a66c', metadata={'doc_number': 5, 'topic': 'RAG'}, page_content='A retriever in a RAG pipeline finds relevant chunks before the language model generates an answer.'), 0.4519554078578949), (Document(id='1ffa50a8-26ec-41b4-9a1f-da1a92cac435', metadata={'topic': 'RAG', 'doc_number': 6}, page_content='Vector stores are important in RAG because they make semantic search over embedded documents possible.'), 0.5010707378387451)]
