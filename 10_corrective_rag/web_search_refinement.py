import re
from pathlib import Path
from typing import TypedDict

from ddgs import DDGS
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from utils.logger import get_logger
from utils.utils import initiate_hf_embedding_model, initiate_llm

logger = get_logger()
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

    web_docs: list[Document]

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

    # combine retrieved good docs into one context string depending upon verdict
    if state["verdict"] == "CORRECT":
        context = "\n\n".join(d.page_content for d in state["good_docs"]).strip()

        # 1. Decomposition: context -> sentence strips
        strips = decompose_to_sentence(context)

        # 2. Filter: keep only relevant strips
        kept: list[str] = []

        for s in strips:
            if filter_chain.invoke({"question": q, "sentence": s}).keep:
                kept.append(s)

        logger.info(
            f"[Internal Docs] Strips extracted: {len(strips)} | Kept: {len(kept)}"
        )
        # 3 Recompse: glue kept strips back together
        refined_context = "\n".join(kept).strip()

        return {
            "strips": strips,
            "kept_strips": kept,
            "refined_context": refined_context,
        }
    else:
        web_docs = state.get("web_docs", [])
        snippets = [d.page_content for d in web_docs if d.page_content.strip()]
        logger.info(f"[Web Search] Retained {len(snippets)} search snippet(s)")

        refined_context = "\n".join(snippets).strip()
        return {
            "strips": snippets,
            "kept_strips": snippets,
            "refined_context": refined_context,
        }


# ---------------------------
# WEB SEARCH
# ---------------------------
def web_search_node(state: State) -> State:
    logger.info("Initiating web search.")
    q = state["question"]

    web_docs: list[Document] = []

    with DDGS() as ddgs:
        results = list(ddgs.text(q, max_results=4))

        if not results:
            logger.warning("No relevant results returned by search engine.")

        for r in results or []:
            title = r.get("title", "") or r.get("heading") or "No Title"
            href = r.get("href", "")
            body = r.get("body", "")

            # Skip generic portal landing pages that lack article substance
            if len(body.strip()) < 40 or "browse thousands of titles" in body.lower():
                continue

            text = f"TITLE: {title}\nHREF: {href}\nBODY: {body}"

            web_docs.append(
                Document(page_content=text, metadata={"href": href, "title": title})
            )

    return {"web_docs": web_docs}


answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant. Synthesize a concise answer to the question using ONLY the provided context snippets.
            If the context is completely empty or contains no relevant information, reply: 'I cannot answer this based on the retrieved information.'""",
        ),
        ("human", "Question: {question}\n\nRefined Context: \n{refined_context}"),
    ]
)


def generate(state: State) -> State:
    logger.info(msg="Generating response")
    logger.debug(f"Refined Context Received:\n{state['refined_context']}")
    out = (answer_prompt | llm).invoke(
        {"question": state["question"], "refined_context": state["refined_context"]}
    )
    # Clean extract if LangChain returns a list of content blocks
    content = out.content
    if isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict):
        content = content[0].get("text", "")
    return {"answer": content}


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
graph = StateGraph(state_schema=State)

# add nodes
graph.add_node("retrieve", retrieve_node)
graph.add_node("eval_each_doc", eval_each_doc_node)
graph.add_node("refine", refine)
graph.add_node("generate", generate)
graph.add_node("web_search", web_search_node)
graph.add_node("ambiguous", ambigouous_node)

# add edges
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "eval_each_doc")

graph.add_conditional_edges(
    "eval_each_doc",
    route_after_eval,
    {"refine": "refine", "web_search": "web_search", "ambiguous": "ambiguous"},
)

# web search also go through refine
graph.add_edge("web_search", "refine")
graph.add_edge("refine", "generate")
graph.add_edge("generate", END)
graph.add_edge("ambiguous", END)

# compile graph
app = graph.compile()

if __name__ == "__main__":
    res: State = app.invoke({"question": "When was the python 3.12 released?"})

    print("Verdict: ", res["verdict"])
    print("Reason: ", res["reason"])
    print("Output: ", res["answer"])

# 12/09/2026 04:06:50 PM INFO     LLM initiated successfully: google_genai:gemini-3.1-flash-lite
# Loading weights: 100%|█████████████████████████████████████████████████████████████| 314/314 [00:00<00:00, 1010.21it/s]
# 12/09/2026 04:07:12 PM INFO     Embedding model initiated successfully: google/embeddinggemma-300m
#                        INFO     Vector store initiated successfully.
#                        INFO     Retriever initiated successfully.
#                        INFO     Retrieving docs.
#                        INFO     Evaluating each docs.
# Direct use of automatic function calling (AFC) in Models.generate_content is not recommended. Instead, we recommend to use AFC in Chat.send_message. Similarly, direct use of AFC in Models.generate_content_stream is not recommended. Instead, we recommend to use AFC in Chat.send_message_stream.
# 12/09/2026 04:07:18 PM INFO     Initiating web search.
# 12/09/2026 04:07:22 PM INFO     Refining docs.
#                        INFO     [Web Search] Retained 4 search snippet(s)
#                        INFO     Generating response
# Verdict:  INCORRECT
# Reason:  All retrieved chunks scored < 0.3. No chunk was sufficient
# Output:  Python 3.12 was released on October 2, 2023.

'''
NOTE: THIS IMPLEMENTATION IS INCOMPLETE
      RESULTS ARE UNDER EXPECTIATIONS WHEN SEARCH RESULT CAME BACK WITH ONLY ADS / TITLES / NOT ENOUGH DATA
      NEEDS TO ADD ANOTHER LLM QUERY TO REFINE WEB RESULTS LIKE BELOW:
        search_prep_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a search query optimizer. 
                    Convert the user's question into an effective web search keyword query.
                    Remove conversational filler. If the question asks for recent events, include the current year/context.
                    Return ONLY the optimized search string.""",
                ),
                ("human", "{question}"),
            ]
        )

        search_prep_chain = search_prep_prompt | llm
'''
