from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from rich.console import Console

load_dotenv()

console = Console()

# colors
primary_col = "yellow"
secondary_col = "cyan"

llm = init_chat_model("google_genai:gemini-3.1-flash-lite", temperature=0)
embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

# Dummy documents covering different topics, each with a mix of relevant and tangential info
docs = [
    Document(
        page_content=(
            "Artificial intelligence has made remarkable strides in natural language processing, "
            "with large language models now capable of generating human-quality text and code. "
            "Computer vision systems can identify objects in images with superhuman accuracy, "
            "powering applications from autonomous vehicles to medical imaging diagnostics. "
            "However, the rapid advancement of AI has raised significant ethical concerns about "
            "job displacement, algorithmic bias, and the concentration of power among a few tech companies."
        ),
        metadata={"topic": "artificial_intelligence"},
    ),
    Document(
        page_content=(
            "Global temperatures have risen by approximately 1.1 degrees Celsius since pre-industrial "
            "times, driven primarily by the burning of fossil fuels. The melting of polar ice caps has "
            "accelerated, contributing to rising sea levels that threaten coastal communities worldwide. "
            "Renewable energy adoption is growing rapidly, with solar and wind power becoming cheaper "
            "than coal in many regions. Governments are implementing carbon pricing mechanisms and "
            "investing in green infrastructure to meet Paris Agreement targets."
        ),
        metadata={"topic": "climate_change"},
    ),
    Document(
        page_content=(
            "NASA's Artemis program aims to return humans to the Moon by the mid-2020s, establishing "
            "a sustainable presence as a stepping stone to Mars. Private companies like SpaceX are "
            "developing reusable rocket technology that has dramatically reduced launch costs. "
            "The James Webb Space Telescope has captured unprecedented images of distant galaxies, "
            "revealing new insights about the early universe. Asteroid mining is being explored as a "
            "potential source of rare minerals needed for electronics manufacturing."
        ),
        metadata={"topic": "space_exploration"},
    ),
    Document(
        page_content=(
            "CRISPR gene editing technology has revolutionized medical genomics, enabling precise "
            "modifications to DNA sequences that were previously impossible. Researchers are using "
            "genomic data to develop personalized medicine approaches, tailoring treatments based on "
            "an individual's genetic profile. Recent breakthroughs in mRNA technology, accelerated by "
            "COVID-19 vaccine development, are now being applied to cancer immunotherapy and rare "
            "genetic disorders. Hospital information systems are increasingly integrating genomic data "
            "to support clinical decision-making at the point of care."
        ),
        metadata={"topic": "medicine"},
    ),
    Document(
        page_content=(
            "The global economy is navigating a period of high inflation driven by supply chain "
            "disruptions, energy price volatility, and post-pandemic demand surges. Central banks "
            "worldwide have raised interest rates aggressively to combat inflation, impacting housing "
            "markets and consumer spending. Cryptocurrency regulation is becoming a priority for "
            "financial authorities, with the EU's MiCA framework setting a global precedent. "
            "Trade tensions between major economies continue to reshape global supply chains, "
            "pushing companies toward nearshoring and diversification strategies."
        ),
        metadata={"topic": "economics"},
    ),
    Document(
        page_content=(
            "Quantum computing has reached a critical milestone with several companies demonstrating "
            "quantum advantage on specific computational tasks. Error correction remains the biggest "
            "challenge, as current quantum processors are highly susceptible to noise and decoherence. "
            "Quantum simulation of molecular structures could transform drug discovery by accurately "
            "modeling protein folding and chemical interactions. Major tech companies and governments "
            "are investing billions in quantum research, viewing it as essential for national security "
            "and economic competitiveness."
        ),
        metadata={"topic": "quantum_computing"},
    ),
]

console.print(f"\nCreated {len(docs)} documents \n", style="blue")

vector_store = InMemoryVectorStore.from_documents(documents=docs, embedding=embedding)

base_retriever = vector_store.as_retriever(search_kwargs={"k": 3})


# Custom LLM Extractor Chain
class CustomLLMExtractorChain:
    """Extracts relevant portions from documents using an LLM chain"""

    def __init__(self, llm_chain):
        self.llm_chain = llm_chain

    @classmethod
    def from_llm(cls, llm):
        prompt = ChatPromptTemplate.from_template(
            "Given the following question and context, extract any part of the context "
            "*AS IS* that is relevant to answer the question. If none of the context is "
            "relevant return NO_OUTPUT.\n\n"
            "Remember, *DO NOT* edit the extracted parts of the context.\n\n"
            "> Question: {question}\n"
            "> Context:\n>>>\n{context}\n>>>\n"
            "Extracted relevant parts:"
        )

        # create chain
        llm_chain = prompt | llm | StrOutputParser()
        return cls(llm_chain=llm_chain)

    def compress_documents(
        self, documents: list[Document], query: str
    ) -> list[Document]:
        compressed = []
        for doc in documents:
            result = self.llm_chain.invoke(
                {"question": query, "context": doc.page_content}
            )
            result = result.strip()
            if result and result != "NO_OUTPUT":
                compressed.append(Document(page_content=result, metadata=doc.metadata))
        return compressed


# custom retriever that composes base retrieval with llm compression
class CustomContextualCompressionRetriever(BaseRetriever):
    """Retriever that compresses documents using a custom LLM extractor chain"""

    base_retriever: BaseRetriever
    base_compressor: Any

    def _get_relevant_documents(self, query: str) -> list[Document]:
        # retrieve documents from base retriever
        docs = self.base_retriever.invoke(input=query)
        # compress using llm extractor
        compressed_docs = self.base_compressor.compress_documents(docs, query)
        return compressed_docs


# run query
compressor = CustomLLMExtractorChain.from_llm(llm=llm)

custom_retriever = CustomContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=base_retriever
)

q = "How is CRISPR acting as a big enabler in creating personalized medicine?"

console.print(f"Query: {q}", style="cyan3")

result = custom_retriever.invoke(input=q)

for i, doc in enumerate(iterable=result, start=1):
    console.print(f"{i}: {doc.metadata['topic']}", style=primary_col)
    console.print(doc.page_content + "\n", style=secondary_col)
