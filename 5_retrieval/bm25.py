from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# 12 documents spanning medicine, architecture, finance, and literature
docs = [
    Document(
        page_content="Antibiotics inhibit bacterial cell wall synthesis or protein production to stop infection.",
        metadata={"topic": "medicine"},
    ),
    Document(
        page_content="Vaccines introduce antigens to train the immune system to recognise and neutralise pathogens.",
        metadata={"topic": "medicine"},
    ),
    Document(
        page_content="MRI scanners use magnetic fields and radio waves to produce detailed images of soft tissue.",
        metadata={"topic": "medicine"},
    ),
    Document(
        page_content="Blood pressure is measured in millimetres of mercury (mmHg) and expressed as systolic over diastolic.",
        metadata={"topic": "medicine"},
    ),
    Document(
        page_content="The Pantheon in Rome was built around 125 AD and still has the world's largest unreinforced concrete dome.",
        metadata={"topic": "architecture"},
    ),
    Document(
        page_content="Gothic cathedrals use flying buttresses to transfer roof weight outward, enabling tall stained glass windows.",
        metadata={"topic": "architecture"},
    ),
    Document(
        page_content="The Bauhaus movement combined fine arts and functional design, influencing modern architecture and typography.",
        metadata={"topic": "architecture"},
    ),
    Document(
        page_content="Compound interest calculates returns on both the initial principal and previously earned interest.",
        metadata={"topic": "finance"},
    ),
    Document(
        page_content="A stock represents partial ownership in a company and entitles the holder to a share of its profits.",
        metadata={"topic": "finance"},
    ),
    Document(
        page_content="Diversification reduces portfolio risk by spreading investments across different asset classes.",
        metadata={"topic": "finance"},
    ),
    Document(
        page_content="Shakespeare wrote 37 plays and 154 sonnets, exploring themes of power, love, and betrayal.",
        metadata={"topic": "literature"},
    ),
    Document(
        page_content="The novel Don Quixote by Cervantes, published in 1605, is often considered the first modern novel.",
        metadata={"topic": "literature"},
    ),
]

# BM25Retriever builds an inverted index from raw document text
# No embedding model is involved — scoring is purely based on token overlap
retriever = BM25Retriever.from_documents(documents=docs, k=2)

# Query 1: exact keyword match — BM25 excels here
# The words "antibiotic" and "bacterial" appear directly in doc 1
q1 = "antibiotic bacterial infection treatment and neutralise pathogens"
r1 = retriever.invoke(input=q1)
print("\nresult for q1")
print(r1)

# result for q1
# [
# Document(metadata={'topic': 'medicine'}, page_content='Vaccines introduce antigens to train the immune system to recognise and neutralise pathogens.'),
# Document(metadata={'topic': 'medicine'}, page_content='Antibiotics inhibit bacterial cell wall synthesis or protein production to stop infection.')]

# Query 2: keyword match across a different topic
# Words like "compound", "interest", and "returns" are present in finance docs

q2 = "compound interest vs Simple interest and what gives a person partial stakes in a company"
r2 = retriever.invoke(input=q2)
print("\nresult for r2")
print(r2)

# [
# Document(metadata={'topic': 'finance'}, page_content='A stock represents partial ownership in a company and entitles the holder to a share of its profits.'),
# Document(metadata={'topic': 'finance'}, page_content='Compound interest calculates returns on both the initial principal and previously earned interest.')]
