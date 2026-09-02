from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from rich.console import Console

load_dotenv()
console = Console()

# colors
primary_col = "yellow"
secondary_col = "cyan"

# create embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
)

# 12 documents spanning health, programming, history, and nature
# Docs 1-2: contain the exact word "vaccine" — BM25 keyword match
# Docs 3-5: semantically related (immune system, antibodies, herd immunity) but lack the word "vaccine"
#            — dense search finds these, BM25 misses them
# Docs 6-12: off-topic
docs = [
    Document(
        page_content="Vaccines work by introducing a weakened or inactivated pathogen to trigger an immune response.",
        metadata={"topic": "health"},
    ),
    Document(
        page_content="The flu vaccine is reformulated each year to match the most prevalent circulating virus strains.",
        metadata={"topic": "health"},
    ),
    Document(
        page_content="The immune system produces antibodies that recognise and neutralise foreign pathogens in the body.",
        metadata={"topic": "health"},
    ),
    Document(
        page_content="Herd immunity occurs when enough of a population becomes resistant to a disease, slowing its spread.",
        metadata={"topic": "health"},
    ),
    Document(
        page_content="White blood cells called B-lymphocytes produce proteins that bind to and destroy specific antigens.",
        metadata={"topic": "health"},
    ),
    Document(
        page_content="Version control systems like Git track changes to code and enable collaboration across teams.",
        metadata={"topic": "programming"},
    ),
    Document(
        page_content="Docker containers package applications with their dependencies for consistent deployment.",
        metadata={"topic": "programming"},
    ),
    Document(
        page_content="The French Revolution began in 1789 and fundamentally transformed European political structures.",
        metadata={"topic": "history"},
    ),
    Document(
        page_content="The Silk Road was an ancient trade network connecting China to the Mediterranean world.",
        metadata={"topic": "history"},
    ),
    Document(
        page_content="The Amazon rainforest produces about 20% of the world's oxygen and houses 10% of all species.",
        metadata={"topic": "nature"},
    ),
    Document(
        page_content="Coral reefs cover less than 1% of the ocean floor but support about 25% of all marine species.",
        metadata={"topic": "nature"},
    ),
    Document(
        page_content="REST APIs communicate over HTTP using standard methods like GET, POST, PUT, and DELETE.",
        metadata={"topic": "programming"},
    ),
]

vector_store = Chroma.from_documents(
    documents=docs, embedding=embeddings, collection_name="hybrid_search"
)

chroma_retriever = vector_store.as_retriever(
    search_type="similarity", search_kwargs={"k": 4}
)

bm25_retriever = BM25Retriever.from_documents(
    documents=docs, k=2, bm25_varient="plus"
)  # plus: works good with small docs

ensemble_retriever = EnsembleRetriever(
    retrievers=[chroma_retriever, bm25_retriever], weights=[0.8, 0.2]
)

q = "How do vaccines work to protect against disease?"

bm25_results = bm25_retriever.invoke(input=q)
chroma_results = chroma_retriever.invoke(input=q)
ensemble_results = ensemble_retriever.invoke(input=q)

console.print(f"\nQuery: {q}", style="blue")
console.print("\nBM25 results", style=primary_col)
for doc in bm25_results:
    console.print(doc.page_content, style=secondary_col)

console.print("\nChroma results", style=primary_col)
for doc in chroma_results:
    console.print(doc.page_content, style=secondary_col)

console.print("\nEnsemble results", style=primary_col)
for doc in ensemble_results:
    console.print(doc.page_content, style=secondary_col)

# BM25 results

# Vaccines work by introducing a weakened or inactivated pathogen to trigger an immune response.
# REST APIs communicate over HTTP using standard methods like GET, POST, PUT, and DELETE.

# Chroma results

# Vaccines work by introducing a weakened or inactivated pathogen to trigger an immune response.
# The immune system produces antibodies that recognise and neutralise foreign pathogens in the body.
# The flu vaccine is reformulated each year to match the most prevalent circulating virus strains.
# White blood cells called B-lymphocytes produce proteins that bind to and destroy specific antigens.

# Ensemble results

# Vaccines work by introducing a weakened or inactivated pathogen to trigger an immune response.
# The immune system produces antibodies that recognise and neutralise foreign pathogens in the body.
# The flu vaccine is reformulated each year to match the most prevalent circulating virus strains.
# White blood cells called B-lymphocytes produce proteins that bind to and destroy specific antigens.
# REST APIs communicate over HTTP using standard methods like GET, POST, PUT, and DELETE.
