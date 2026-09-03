from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
    LLMChainExtractor,
)
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from rich.console import Console

load_dotenv()

console = Console()

# colors
primary_col = "yellow"
secondary_col = "cyan"

# create embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

# create llm
llm = init_chat_model("google_genai:gemini-3.1-flash-lite", temperature=0)

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

console.print(f"Created {len(docs)} documents \n", style="blue")

# create vector store
vector_store = InMemoryVectorStore.from_documents(documents=docs, embedding=embeddings)

# base retriever
base_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# retrieval without compression
q = "How is CRISPR acting as a big enabler in creating personalized medicine?"

console.print(f"Query: {q}", style="cyan3")

base_results = base_retriever.invoke(input=q)

for i, doc in enumerate(iterable=base_results, start=1):
    console.print(f"{i}: {doc.metadata['topic']}", style=primary_col)
    console.print(doc.page_content + "\n", style=secondary_col)

# Query: How is CRISPR acting as a big enabler in creating personalized medicine?
# 1: medicine
# CRISPR gene editing technology has revolutionized medical genomics, enabling precise modifications to DNA sequences that
# were previously impossible. Researchers are using genomic data to develop personalized medicine approaches, tailoring
# treatments based on an individual's genetic profile. Recent breakthroughs in mRNA technology, accelerated by COVID-19
# vaccine development, are now being applied to cancer immunotherapy and rare genetic disorders. Hospital information
# systems are increasingly integrating genomic data to support clinical decision-making at the point of care.

# 2: quantum_computing
# Quantum computing has reached a critical milestone with several companies demonstrating quantum advantage on specific
# computational tasks. Error correction remains the biggest challenge, as current quantum processors are highly
# susceptible to noise and decoherence. Quantum simulation of molecular structures could transform drug discovery by
# accurately modeling protein folding and chemical interactions. Major tech companies and governments are investing
# billions in quantum research, viewing it as essential for national security and economic competitiveness.

# 3: space_exploration
# NASA's Artemis program aims to return humans to the Moon by the mid-2020s, establishing a sustainable presence as a
# stepping stone to Mars. Private companies like SpaceX are developing reusable rocket technology that has dramatically
# reduced launch costs. The James Webb Space Telescope has captured unprecedented images of distant galaxies, revealing
# new insights about the early universe. Asteroid mining is being explored as a potential source of rare minerals needed
# for electronics manufacturing.

# LLMChainExtractor uses an llm to extract only the relevant portions from each document
compressor = LLMChainExtractor.from_llm(llm=llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=base_retriever
)

compressed_results = compression_retriever.invoke(input=q)

console.print("Compressed Results", style="cyan3")

for i, doc in enumerate(iterable=compressed_results, start=1):
    console.print(f"{i}: {doc.metadata['topic']}", style=primary_col)
    console.print(doc.page_content + "\n", style=secondary_col)


# Compressed Results
# 1: medicine
# Extracted relevant parts: CRISPR gene editing technology has revolutionized medical genomics, enabling precise
# modifications to DNA sequences that were previously impossible. Researchers are using genomic data to develop
# personalized medicine approaches, tailoring treatments based on an individual's genetic profile.

# embedding filter: filter with threshold
embedding_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.7)

embedding_filter_retriever = ContextualCompressionRetriever(
    base_compressor=embedding_filter, base_retriever=base_retriever
)

embedding_results = embedding_filter_retriever.invoke(input=q)

console.print("Embedding filter Results", style="cyan3")
for i, doc in enumerate(iterable=embedding_results, start=1):
    console.print(f"{i}: {doc.metadata['topic']}", style=primary_col)
    console.print(doc.page_content + "\n", style=secondary_col)

# Embedding filter Results
# 1: medicine
# CRISPR gene editing technology has revolutionized medical genomics, enabling precise modifications to DNA sequences that
# were previously impossible. Researchers are using genomic data to develop personalized medicine approaches, tailoring
# treatments based on an individual's genetic profile. Recent breakthroughs in mRNA technology, accelerated by COVID-19
# vaccine development, are now being applied to cancer immunotherapy and rare genetic disorders. Hospital information
# systems are increasingly integrating genomic data to support clinical decision-making at the point of care.

# Document Compressor Pipeline: chain multiple compressors together
# first filter by embeddings then extract with llm -> in embedding filter results we seen that it reduces the noise by removing non related docs e.g quantum computing and space exploration but it was unable to go inside the chunk and slice the related content only, it returns the whole chunk, in llm compressed results, it gave us the desired result but we had to send 3 chunks to document which is expensive

pipeline_compressor = DocumentCompressorPipeline(
    transformers=[embedding_filter, compressor]  # transformer order is important
)

pipeline_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor, base_retriever=base_retriever
)

pipeline_results = pipeline_retriever.invoke(input=q)

console.print("Pipeline Results", style="cyan3")
for i, doc in enumerate(iterable=pipeline_results, start=1):
    console.print(f"{i}: {doc.metadata['topic']}", style=primary_col)
    console.print(doc.page_content + "\n", style=secondary_col)

# Pipeline Results
# 1: medicine
# Extracted relevant parts:
# CRISPR gene editing technology has revolutionized medical genomics, enabling precise modifications to DNA sequences that
# were previously impossible. Researchers are using genomic data to develop personalized medicine approaches, tailoring
# treatments based on an individual's genetic profile.
