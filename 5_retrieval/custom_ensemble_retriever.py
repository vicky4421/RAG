from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
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

# Same 12 documents as the hybrid search notebook
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
    documents=docs, embedding=embeddings, collection_name="custom_ensemble"
)

chroma_retriever = vector_store.as_retriever(
    search_type="similarity", search_kwargs={"k": 4}
)

bm25_retriever = BM25Retriever.from_documents(documents=docs, k=2, bm25_varient="plus")


class CustomEnsembleRetrieval(BaseRetriever):
    """
    Custom ensemble retriever that fuses results from multiple retrievers
    using Reciprocal Rank Fusion (RRF).

    RRF score for a document d across retriever i:
        score(d) = sum over i of [ weight_i * (1 / (rank_i(d) + rrf_k)) ]

    rrf_k is a smoothing constant (default 60) — it dampens the outsized
    advantage of rank-1 documents so lower-ranked results still contribute.
    Documents not returned by a retriever contribute 0 for that retriever.
    """

    retrievers: list[BaseRetriever]
    weights: list[float]
    rrf_k: int = 60

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        # collect ranked result list from every retriever
        all_results: list[list[Document]] = [
            retriever.invoke(query) for retriever in self.retrievers
        ]

        # accumulate RRf scores
        doc_scores: dict[str, tuple[float, Document]] = {}

        for retriever_idx, results in enumerate(iterable=all_results):
            weight = self.weights[retriever_idx]
            for rank, doc in enumerate(results):
                rrf_score = weight * (1.0 / (rank + self.rrf_k))
                key = doc.page_content
                if key in doc_scores:
                    prev_score, prev_doc = doc_scores[key]
                    doc_scores[key] = (prev_score + rrf_score, prev_doc)
                else:
                    doc_scores[key] = (rrf_score, doc)

        sorted_docs = sorted(doc_scores.values(), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in sorted_docs]


ensemble_retriever = CustomEnsembleRetrieval(
    retrievers=[chroma_retriever, bm25_retriever], weights=[0.8, 0.2], rrf_k=60
)

q = "How do vaccine protect from disease?"

bm25_results = bm25_retriever.invoke(q)
chroma_results = chroma_retriever.invoke(q)
ensemble_results = ensemble_retriever.invoke(q)

console.print(f"\nQuery: {q}", style="blue")
console.print("\nBM25 results", style=primary_col)
for i, doc in enumerate(iterable=bm25_results, start=1):
    console.print(f"{i} {doc.page_content}", style=secondary_col)

console.print("\nChroma results", style=primary_col)
for i, doc in enumerate(iterable=chroma_results, start=1):
    console.print(f"{i} {doc.page_content}", style=secondary_col)

console.print("\nEnsemble results", style=primary_col)
for i, doc in enumerate(iterable=ensemble_results, start=1):
    console.print(f"{i} {doc.page_content}", style=secondary_col)

# Query: How do vaccine protect from disease?

# BM25 results
# 1 The flu vaccine is reformulated each year to match the most prevalent circulating virus strains.
# 2 REST APIs communicate over HTTP using standard methods like GET, POST, PUT, and DELETE.

# Chroma results
# 1 Vaccines work by introducing a weakened or inactivated pathogen to trigger an immune response.
# 2 The immune system produces antibodies that recognise and neutralise foreign pathogens in the body.
# 3 Herd immunity occurs when enough of a population becomes resistant to a disease, slowing its spread.
# 4 The flu vaccine is reformulated each year to match the most prevalent circulating virus strains.

# Ensemble results
# 1 The flu vaccine is reformulated each year to match the most prevalent circulating virus strains.
# 2 Vaccines work by introducing a weakened or inactivated pathogen to trigger an immune response.
# 3 The immune system produces antibodies that recognise and neutralise foreign pathogens in the body.
# 4 Herd immunity occurs when enough of a population becomes resistant to a disease, slowing its spread.
# 5 REST APIs communicate over HTTP using standard methods like GET, POST, PUT, and DELETE.
