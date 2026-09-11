import re
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from rich.console import Console

from utils.logger import get_logger
from utils.utils import initiate_hf_embedding_model, initiate_llm

load_dotenv()
console = Console()
logger = get_logger()

# chroma db path (move to root in next chapter)
chroma_path = Path(__file__).resolve().parent / "chroma_db"

logger.info(msg="Pipeline initiated.")

# llm = init_chat_model("google_genai:gemini-3.1-flash-lite")
llm = initiate_llm()

embeddings = initiate_hf_embedding_model(
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
)

vector_store = Chroma(
    collection_name="corrective_rag_demo",
    embedding_function=embeddings,
    persist_directory=str(chroma_path),
)
logger.info("Vector store initiated successfully.")

retriver = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
logger.info("Retriever initiated successfully.")


# define state schema
class State(TypedDict):
    question: str
    docs: list[Document]
    answer: str

    strips: list[str]  # output of decomposition (sentence strips)
    kept_strips: list[str]  # after filtering (kept sentences)
    refined_context: str  # recomposed internal knowledge (joined kept_strips)


def retrieve(state: State) -> State:
    logger.info("Retriving docs.")
    q = state["question"]
    return {"docs": retriver.invoke(input=q)}


# ---------------------------
# Sentence level decomposer
# ---------------------------
def decompose_to_sentence(text: str) -> list[str]:
    logger.info("Decomposing to sentences.")
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


# ---------------------------
# FILTER (llm judge)
# ---------------------------


class KeepOrDrop(BaseModel):
    keep: bool


filter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You're a strict relevance filter\n
            Return keep=true only if the sentence directly helps answer the question\n
            Use ONLY sentence. Output JSON only""",
        ),
        ("human", "Question: {question}\n\nSentence: \n{sentence}"),
    ]
)

filter_chain = filter_prompt | llm.with_structured_output(schema=KeepOrDrop)


# ---------------------------
# REFINING (Decompose -> Filter -> Recompose)
# ---------------------------
def refine(state: State) -> State:
    logger.info("Refining response.")
    q = state["question"]

    # combine retrieved docs into one context string
    context = "\n\n".join(d.page_content for d in state["docs"]).strip()

    # 1. Decomposition: context -> sentence strips
    strips = decompose_to_sentence(context)

    # 2. Filter: keep only relevant strips
    kept: list[str] = []

    for s in strips:
        if filter_chain.invoke({"question": q, "sentence": s}).keep:
            kept.append(s)

    # 3 Recompse: glue kept strips back together
    refined_context = "\n".join(kept).strip()

    return {"strips": strips, "kept_strips": kept, "refined_context": refined_context}


answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You're a helpful ML tutor. Answer only using the provided refined bullets.\n
            If the bullets are empty or insufficient, say: 'I don't know based on the provided books'""",
        ),
        ("human", "Question: {question}\n\nRefined Context: \n{refined_context}"),
    ]
)


def generate(state: State) -> State:
    logger.info(msg="Generating response")
    out = (answer_prompt | llm).invoke(
        {"question": state["question"], "refined_context": state["refined_context"]}
    )
    return {"answer": out.content}


graph = StateGraph(state_schema=State)

# add nodes
graph.add_node("retrieve", retrieve)
graph.add_node("refine", refine)
graph.add_node("generate", generate)

# add edges
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "refine")
graph.add_edge("refine", "generate")
graph.add_edge("generate", END)

# compile graph
app = graph.compile()

# final response
response: State = app.invoke({"question": "Explain the bias variance tradeoff"})
print(response["kept_strips"])
print()
print(response["answer"])


# ['Learning curves for the polynomial model One way to improve an overfitting model is to feed it more training da ta until the validation error reaches the training error.',
# 'The Bias/Variance Tradeoff An importan t theoretical result of statistics and Machine Learning is the fact that a model’s generaliza tion error can be expressed as the sum of three very different errors: Bias This part of the generalization error is due to wrong assumptions, such as assum‐ ing that the data is linear when it is actually quadratic.',
# 'A high-bias model is most likely to underfit the training data.10 Variance This part is due to the model’s excessive sensitivity to small variations in the training data.',
# 'Increasing a model’s complexity will typically increase its variance and reduce its bias.',
# 'Conversely, reducing a model’s complexity increases its bias and reduces its variance.',
# 'This is why it is called a tradeoff.',
# 'Regularized Linear Models As we saw in Chapters 1 and 2, a good way to reduce overfitting is to regularize the model (i.e., to constrain it): the fewer degrees of freedom it has, the harder it will be for it to overfit the data.',
# 'For example, a simple way to regularize a polynomial model is to reduce the number of polynomial degrees.',
# 'Suppose you are using Ridge Regression and you notice that the training error and the validation error are almost equal and fairly high.',
# 'Would you sa y that the model suffers from high bias or high variance?',
# 'Should you increase the regulari‐ zation hyperparameter α or reduce it?']

# [{'type': 'text', 'text': 'A model’s generalization error is the sum of three different errors, including bias and variance:\n\n*   **Bias:** Error due to wrong assumptions (e.g., assuming data is linear when it is quadratic). A high-bias model is likely to underfit the training data.\n*   **Variance:** Error due to the model’s excessive sensitivity to small variations in the training data.\n*   **The Tradeoff:** Increasing a model’s complexity typically increases its variance and reduces its bias. Conversely, reducing a model’s complexity increases its bias and reduces its variance.', 'extras': {'signature': 'EnEKbwERTTIPo4Bic2t0hGirjA6c+xoHKMOF5hgSMFvjdSUt7FW2iGJd+vnSYrRnRs0ih73PrRYkDrY7LeBhKP43o+NaqLymhSnNKzF+Q1ao1lmR4RYvkF0AwGno+w0RrfbE4N+/Ro2QycbjU2QwsUNXgw=='}}]
