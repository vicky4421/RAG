import logging
import os

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables.base import Runnable
from pydantic import BaseModel, Field

# create log file in current file path
_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_fusion.log")

logger = logging.getLogger("rag_fusion_logger")
logger.setLevel(level=logging.INFO)

# if no handler
if not logger.handlers:
    _file_handler = logging.FileHandler(_log_path)
    _file_handler.setFormatter(
        fmt=logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(messages)s")
    )
    logger.addHandler(_file_handler)


# schema
class SubQuerySchema(BaseModel):
    sub_queries: list[str] = Field(
        description="List of sub queries to be generated from the main query"
    )


class RAGFusion:
    """
    RAG Fusion is a retrieval-augmented generation (RAG) system that combines
    multiple-retrievers or Sub-Queries to enhance the retrieval process. It uses an ensemble of
    retrievers or Sub-Queries to retrieve relevant documents and then fuses the results to generate
    a final response.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm_chain: Runnable | None = None,
        num_subqueries: int = 3,
        k: int = 5,
    ):
        self.retriever = retriever
        self.llm_chain = llm_chain
        self.num_subqueries = num_subqueries
        self.k = k

    # class method if retrievers are provided i.e. ensemble
    @classmethod
    def from_retrievers(
        cls,
        base_retrievers: list[BaseRetriever],
        weights: list[float] | None = None,
        k: int = 5,
    ):
        if not base_retrievers:
            raise ValueError("At least one retriever must be provided")

        if not isinstance(base_retrievers, list):
            raise TypeError("base retrievers must be a list of retriever instances")

        ensemble = EnsembleRetriever(retrievers=base_retrievers, weights=weights)

        return cls(retriever=ensemble, k=k)

    # class method if single query provided and not multiple retrievers
    @classmethod
    def from_llm(
        cls, llm, retriever: BaseRetriever, num_subqueries: int = 3, k: int = 5
    ):
        # create prompt template to generate sub queries
        prompt = ChatPromptTemplate(
            messages=[
                (
                    "system",
                    "You are a helpful assistant that generates sub-queries from a main query to enhance retrieval.",
                ),
                (
                    "user",
                    "Given the main query: '{main_query}', generate {num_subqueries} sub-queries that can be used to retrieve relevant documents.",
                ),
            ],
            input_variables=["main_query", "num_subqueries"],
        )

        # use the llm with structured output to generate sub-queries
        structured_llm = llm.with_structured_output(schema=SubQuerySchema)

        # create chain
        llm_chain = prompt | structured_llm

        return cls(retriever, llm_chain, num_subqueries, k)

    # internal function to retrieve documents
    def _retrieve_documents(self, query: str) -> list[Document]:
        return self.retriever.invoke(input=query)

    # internal function to generate subqueries using llm
    def _generate_subqueries(self, query: str) -> list[str]:
        if not self.llm_chain:
            raise ValueError("LLM chain is not provided to generate sub queries")

        logger.info(
            msg=f"Generating {self.num_subqueries} subqueries for main query: {query}"
        )

        result = self.llm_chain.invoke(
            {"main_query": query, "num_subqueries": self.num_subqueries}
        )

        logger.info(f"Generated subqueries: {result.sub_queries}")
        return result.sub_queries

    # internal method to perform RRF
    def _reciprocal_rank_fusion(
        self, retrieved_docs: list[list[Document]]
    ) -> list[Document]:

        doc_scores: dict[str, tuple[float, Document]] = {}

        for retrieved_set in retrieved_docs:
            for rank, doc in enumerate(retrieved_set, start=1):
                # calculate rrf score
                rrf_score = 1 / (rank + 60)

                # fetch doc content
                key = doc.page_content

                # if doc content is already in doc_scores, update its score
                if key in doc_scores:
                    # fetch prev score and document form doc_scores
                    prev_score, prev_doc = doc_scores[key]

                    # update prev score by adding rrf score and update rrf dict
                    doc_scores[key] = (prev_score + rrf_score, prev_doc)

                # if doc content is not in doc score, add the document to doc_score
                else:
                    doc_scores[key] = (rrf_score, doc)

        # sort documents wrt new scores
        doc_with_scores = doc_scores.values()
        sorted_docs = sorted(
            doc_with_scores, key=lambda x: x[0], reverse=True
        )  # x[0]: rrf score

        logger.info(f"RRF scores with documents: {doc_with_scores}")

        return [doc for _, doc in sorted_docs]

    # method to invoke rag fusion process
    def invoke(self, query: str) -> list[Document]:
        if self.llm_chain:
            sub_queries = self._generate_subqueries(query=query)
            all_retrived_docs = [
                self._retrieve_documents(subquery) for subquery in sub_queries
            ]
            logger.info(
                f"Retrieved documents per sub-query: {[[doc.page_content[0:50] for doc in docs] for docs in all_retrived_docs]}"
            )
            fused_docs = self._reciprocal_rank_fusion(retrieved_docs=all_retrived_docs)
            return fused_docs[: self.k]

        else:
            return self._retrieve_documents(query=query)[: self.k]
