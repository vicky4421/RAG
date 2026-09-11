import re
from pathlib import Path
from typing import TypedDict

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from utils.logger import get_logger
from utils.utils import initiate_hf_embedding_model, initiate_llm

logger = get_logger()

logger.info("Pipeline initiated.")

chroma_path = Path(__file__).resolve().parent / "chroma_db"

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

# define thresholds to filter docs
UPPER_THRESHOLD = 0.7
LOWER_THRESHOLD = 0.3


# define state schema
class State(TypedDict):
    question: str
    docs: list[Document]

    good_docs: list[Document]
    verdict: str
    reason: str

    strips: list[str]  # output of decomposition (sentence strips)
    kept_strips: list[str]  # after filtering (kept sentences)
    refined_context: str  # recomposed internal knowledge (joined kept_strips)

    answer: str


def retrieve_node(state: State) -> State:
    logger.info("Retrieving docs.")
    q = state["question"]
    return {"docs": retriver.invoke(input=q)}


# ---------------------------
# SCORE BASED doc evaluatore
# ---------------------------
class Doc_Eval_Score(BaseModel):
    score: float
    reason: str


doc_eval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You're a strict retrieval evaluator for RAG\n
                You'll be given one retrieved chunk and a question\n
                Return a relevance score in [0.0-1.0]\n
                - 1.0: chunk alone is sufficient to answer fully/mostly\n
                - 0.0: chunk is irrelevant\n
                Be conservative with high scores\n
                Also return a short reason\n
                Output json only""",
        ),
        ("human", "Question: {question}\n\nChunk: \n{chunk}"),
    ]
)

doc_eval_chain = doc_eval_prompt | llm.with_structured_output(schema=Doc_Eval_Score)


def eval_each_doc_node(state: State) -> State:
    logger.info("Evaluating each docs.")
    q = state["question"]

    scores: list[float] = []
    reason: list[str] = []
    good: list[Document] = []

    for d in state["docs"]:
        out: Doc_Eval_Score = doc_eval_chain.invoke(
            {"question": q, "chunk": d.page_content}
        )
        scores.append(out.score)
        reason.append(out.reason)

        if out.score > LOWER_THRESHOLD:
            good.append(d)

    if any(s > UPPER_THRESHOLD for s in scores):
        return {
            "good_docs": good,
            "verdict": "CORRECT",
            "reason": f"at least one retrieved chunk score > {UPPER_THRESHOLD}",
        }

    if len(scores) > 0 and all(s < LOWER_THRESHOLD for s in scores):
        why = "No chunk was sufficient"
        return {
            "good_docs": [],
            "verdict": "INCORRECT",
            "reason": f"All retrieved chunks scored < {LOWER_THRESHOLD}. {why}",
        }

    why = "Mixed relevance signals"
    return {
        "good_docs": good,
        "verdict": "AMBIGUOUS",
        "reason": f"No chunk scored > {UPPER_THRESHOLD}, but not all were < {LOWER_THRESHOLD}. {why}",
    }


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
    logger.info("Refining docs.")
    q = state["question"]

    # combine retrieved good docs into one context string
    context = "\n\n".join(d.page_content for d in state["good_docs"]).strip()

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


def fail_node(state: State) -> State:
    logger.info("Failed to retrieve useful docs.")
    return {"answer": f"FAIL: {state['reason']}"}


def ambigouous_node(state: State) -> State:
    logger.info("Retrieved docs are ambiguous.")
    return {"answer": f"AMBIGOUOUS: {state['reason']}"}


def route_after_eval(state: State) -> str:
    if state["verdict"] == "CORRECT":
        return "refine"
    elif state["verdict"] == "INCORRECT":
        return "web_search"
    else:
        return "AMBIGUOUS"


# define graph
workflow = StateGraph(state_schema=State)

# add nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("eval_each_doc", eval_each_doc_node)
workflow.add_node("refine", refine)
workflow.add_node("generate", generate)
workflow.add_node("fail", fail_node)
workflow.add_node("ambiguous", ambigouous_node)

# add edges
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "eval_each_doc")

workflow.add_conditional_edges(
    "eval_each_doc",
    route_after_eval,
    {"refine": "refine", "web_search": "fail", "ambiguous": "ambiguous"},
)

workflow.add_edge("refine", "generate")
workflow.add_edge("generate", END)
workflow.add_edge("fail", END)

# compile graph
graph = workflow.compile()

res: State = graph.invoke({"question": "What is bias variance tradeoff?"})

print("Verdict: ", res["verdict"])
print("Reason: ", res["reason"])
print("Output: ", res["answer"])

# What are attention mechanisms and why are they important in current models?
# Verdict:  INCORRECT
# Reason:  All retrieved chunks scored < 0.3. No chunk was sufficient
# Output:  FAIL: All retrieved chunks scored < 0.3. No chunk was sufficient

# Verdict:  CORRECT
# Reason:  at least one retrieved chunk score > 0.7
# Output:  [{'type': 'text', 'text': 'The bias/variance tradeoff is an important theoretical result stating that a model’s generalization error is the sum of three errors, including bias and variance:\n\n*   **Bias:** This error arises from wrong assumptions (e.g., assuming linear data when it is quadratic). High-bias models are likely to underfit the training data.\n*   **Variance:** This error is due to a model’s excessive sensitivity to small variations in the training data.\n*   **The Tradeoff:** Increasing a model’s complexity typically increases its variance and reduces its bias, while reducing a model’s complexity increases its bias and reduces its variance.', 'extras': {'signature': 'EnEKbwERTTIP+kBDVNN9apKC8Dodhxeb0n0GOdfFHHMaDP//4WraaaSAUTKdRwk/2dufB11GWUh/Xqlwe04WOosU5nDE/m0ExaCIBHyUlN+ds0g2hx6uehOK1b11NQNnjqM5ZEBvP30jP26yRxrY4AVNIw=='}}]
