from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# get project root path
project_root = Path(__file__).parent.parent

pdf_path = (
    project_root / "4_vector_store" / "beyond-chatbots-ai-agents-next-real-shift.pdf"
)

collection_name = "chroma_pipeline"

# using same persist directory
persist_directory = project_root / "4_vector_store" / "chroma_db"

# create embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
)


# helper function to print document
def print_docs(title: str, docs: Document):
    """Print retrieved documents using page metadata and a text preview."""
    print(title)
    for index, doc in enumerate(docs, start=1):
        print(
            f"{index}. page={doc.metadata.get('page')} | source={doc.metadata.get('source')}"
        )
        print(f"   content={doc.page_content}")
    print()


# Load PDF
pdf_loader = PyPDFLoader(file_path=pdf_path)

docs = pdf_loader.load()
print(f"Total pages loaded from pdf: {len(docs)}")

# Split PDF
pdf_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)

chunked_docs = pdf_splitter.split_documents(documents=docs)
print(f"Total chunks generated: {len(chunked_docs)}")

vector_store = Chroma.from_documents(
    documents=chunked_docs,
    embedding=embeddings,
    collection_name=collection_name,
    persist_directory=str(persist_directory),
)
print(f"Stored {len(chunked_docs)} chunks in the collection: {collection_name}")

# Retrieve relevant chunks
query = "How do AI agents use tools and memory?"

result = vector_store.similarity_search(query=query, k=3)

print(f"Query: {query}\n")
print_docs(title="Retrieved Chunks\n", docs=result)
