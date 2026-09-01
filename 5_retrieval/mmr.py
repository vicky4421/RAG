from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# 15 documents: the first 3 deep learning docs are intentionally near-identical —
# all describe gradient descent as the core optimisation technique, with minor phrasing variation.
# The next 3 deep learning docs cover distinct regularisation/training concepts.
# At lambda_mult=1.0 (pure relevance), a query about deep learning training returns
# all 3 near-identical gradient descent docs, showing the redundancy problem.
# As lambda_mult decreases, MMR penalises already-selected similar docs and
# picks one gradient descent doc + Dropout + Batch Norm + LR Scheduler instead.
docs = [
    Document(
        page_content="Training a deep learning model involves iteratively adjusting weights using gradient descent to minimise the loss.",
        metadata={"topic": "deep learning"},
    ),
    Document(
        page_content="Deep learning models are optimised through gradient descent, which updates weights in the direction that reduces the training loss.",
        metadata={"topic": "deep learning"},
    ),
    Document(
        page_content="Gradient descent is the core optimisation technique in deep learning, guiding weight updates based on computed gradients of the loss.",
        metadata={"topic": "deep learning"},
    ),
    Document(
        page_content="Dropout randomly disables a fraction of neurons during training to prevent overfitting in deep networks.",
        metadata={"topic": "deep learning"},
    ),
    Document(
        page_content="Batch normalisation stabilises training by normalising layer inputs, which allows the use of higher learning rates.",
        metadata={"topic": "deep learning"},
    ),
    Document(
        page_content="Learning rate schedulers dynamically adjust the learning rate during training to improve convergence and avoid overshooting.",
        metadata={"topic": "deep learning"},
    ),
    Document(
        page_content="Arctic sea ice has declined by about 13% per decade since satellite measurements began in 1979.",
        metadata={"topic": "climate"},
    ),
    Document(
        page_content="Carbon capture technology removes CO2 from the atmosphere and stores it underground.",
        metadata={"topic": "climate"},
    ),
    Document(
        page_content="The permafrost in Siberia contains vast amounts of methane that could be released as it thaws.",
        metadata={"topic": "climate"},
    ),
    Document(
        page_content="The Renaissance was a cultural movement in Europe from the 14th to 17th century that revived classical art.",
        metadata={"topic": "art"},
    ),
    Document(
        page_content="Impressionism emerged in 19th-century France, focusing on light, colour, and everyday subjects.",
        metadata={"topic": "art"},
    ),
    Document(
        page_content="Abstract expressionism prioritises spontaneous, automatic, and subconscious creation.",
        metadata={"topic": "art"},
    ),
    Document(
        page_content="Common law systems derive legal principles from judicial precedent rather than written codes.",
        metadata={"topic": "law"},
    ),
    Document(
        page_content="The presumption of innocence requires the prosecution to prove guilt beyond reasonable doubt.",
        metadata={"topic": "law"},
    ),
    Document(
        page_content="Intellectual property law protects creations of the mind, including patents, trademarks, and copyrights.",
        metadata={"topic": "law"},
    ),
]

# create embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
)

# InMemory vector store
vector_store = Chroma.from_documents(
    documents=docs, embedding=embeddings, collection_name="mmr_demo"
)

q = "deep learning model training and its optimization techniquies"

# lambda_mult controls the relevance-diversity trade-off:
#   1.0 = pure relevance (identical to similarity search)
#   0.0 = pure diversity (ignores relevance entirely)
# fetch_k: number of candidate docs fetched before MMR re-ranks and selects k

retriever = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5}
)

result = retriever.invoke(input=q)

print(result)

# [
# Document(id='cf76f007-2118-4d10-8a19-241f4f475bac', metadata={'topic': 'deep learning'}, page_content='Deep learning models are optimised through gradient descent, which updates weights in the direction that reduces the training loss.'),
# Document(id='b1fe896a-12ea-4bc6-845d-c22b44f35e06', metadata={'topic': 'deep learning'}, page_content='Learning rate schedulers dynamically adjust the learning rate during training to improve convergence and avoid overshooting.'),
# Document(id='e13c4cdd-f569-4eb3-bfc5-76434f10fd05', metadata={'topic': 'deep learning'}, page_content='Dropout randomly disables a fraction of neurons during training to prevent overfitting in deep networks.')]
