import asyncio
import logging
import os
import sys
from unittest.mock import MagicMock

# Intercept the dead import BEFORE any ragas imports run:
sys.modules["langchain_community.chat_models.vertexai"] = MagicMock()

from typing import Any

from openai import AsyncOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory

from ch_13_ragas.rag_pipeline import chain, retriever

# silent gemini warnings
logging.getLogger(name="google_genai.models").setLevel(level=logging.ERROR)

# (question, ground_truth_reference)
QA_PAIRS = [
    (
        "What does sustainable development mean?",
        "Development that meets today's needs without preventing future generations from meeting their own needs",
    ),
    (
        "What are the three pillars of sustainable development?",
        "Economic growth, social inclusion, and environmental protection",
    ),
    (
        "How many global goals did countries agree on in the 2030 Agenda?",
        "17 Sustainable Development Goals",
    ),
    (
        "What is the Paris Agreement's temperature target?",
        "Limit global warming to 1.5 degrees Celsius above pre-industrial levels",
    ),
    (
        "What does a circular economy try to reduce?",
        "Waste, by keeping products and materials in use for as long as possible",
    ),
    (
        "What are two examples of circular economy actions?",
        "Recycling and repairing products instead of throwing them away",
    ),
    (
        "What basic needs are highlighted under social equity?",
        "Access to clean water, basic education, and healthcare for all people",
    ),
    (
        "What role do individuals play in sustainable development?",
        "Individuals can reduce waste, save energy, and make eco-friendly choices",
    ),
    (
        "Why is limiting global warming to 1.5 degrees Celsius important?",
        "Beyond 1.5 degrees Celsius, risks of floods, droughts, and extreme weather increase significantly",
    ),
    (
        "What is the target year for the Sustainable Development Goals?",
        "2030",
    ),
]


# Helper to guarantee a clean string regardless of LangChain's output format
def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Handles [{'type': 'text', 'text': '...'}, ...]
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)


def run_evaluation(chain, retriever):

    google_client = AsyncOpenAI(
        api_key=os.getenv(key="GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        max_retries=5,
    )

    google_embedding = embedding_factory(
        provider="openai", model="gemini-embedding-2-preview", client=google_client
    )

    hf_embedding_gemma = embedding_factory(
        provider="huggingface",
        model=r"C:\ai_models_cache\huggingface\models--google--embeddinggemma-300m\snapshots\57c266a740f537b4dc058e1b0cda161fd15afa75",
        device="cpu",
        normalize_embeddings=True,
        batch_size=32,
    )

    evaluator_llm = llm_factory(model="gemini-3.1-flash-lite", client=google_client)

    dataset: list[dict[str, Any]] = []

    for i, (question, ground_truth) in enumerate(QA_PAIRS):
        context_docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in context_docs]

        context_str = "\n\n".join(contexts)
        result = chain.invoke({"context": context_str, "question": question})
        response = extract_text(result.content)

        dataset.append(
            {
                "user_input": question,
                "retrieved_contexts": contexts,
                "response": response,
                "reference": ground_truth,
            }
        )

    evaluation_dataset = EvaluationDataset.from_list(dataset)

    result = evaluate(
        dataset=evaluation_dataset, embeddings=hf_embedding_gemma, llm=evaluator_llm
    )

    df = result.to_pandas()

    # save to csv
    df.to_csv("evaluation_results.csv", index=False)


if __name__ == "__main__":
    run_evaluation(chain, retriever)
