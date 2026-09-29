import asyncio
import os
import sys
from unittest.mock import MagicMock

# Intercept the dead import BEFORE any ragas imports run:
sys.modules["langchain_community.chat_models.vertexai"] = MagicMock()

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy

from utils.logger import get_logger

load_dotenv()
logger = get_logger(log_name="ragas.log")

model_3_6 = "gemini-3.6-flash"
model_3_1 = "gemini-3.1-flash-lite"
genai_model = model_3_1

# google_client = genai.Client(api_key=os.getenv(key="GEMINI_API_KEY"))
google_client = AsyncOpenAI(
    api_key=os.getenv(key="GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    max_retries=5,
)

llm = llm_factory(model=genai_model, client=google_client)

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
logger.info("Embedding model ready.")

embedding = hf_embedding_gemma

scorer = AnswerRelevancy(llm=llm, embeddings=embedding)


async def main():
    # logger.info(
    #     "# Example 1 : Response answers the question but drifts into tangential information"
    # )

    # result = await scorer.ascore(
    #     user_input="What is the capital of Australia?",
    #     response="Australia is a large country in the Southern Hemisphere. It has many major cities including Sydney, Melbourne, and Brisbane. Canberra is the capital city, chosen as a compromise between Sydney and Melbourne. Australia also has a diverse economy driven by mining and agriculture.",
    # )

    # logger.info(
    #     f"Noise sensativity score with model {genai_model} and {embedding}: {result.value}"
    # )

    # logger.info(
    #     "# Example 2: Response is direct and precisely answers the question with no filler"
    # )

    # result = await scorer.ascore(
    #     user_input="When was the first Super Bowl played?",
    #     response="The first Super Bowl was played on January 15, 1967.",
    # )

    # logger.info(
    #     f"Noise sensativity score with model {genai_model} and {embedding}: {result.value}"
    # )

    logger.info(
        "# Example 3 : Response talks about water as a topic but never states its boiling point"
    )

    result = await scorer.ascore(
        user_input="What is the boiling point of water?",
        response="Water is a fascinating substance found all over the Earth. It is essential for all known forms of life and covers about 71 percent of the Earth's surface. Water is found in oceans, rivers, lakes, and glaciers and plays a key role in regulating climate.",
    )

    logger.info(
        f"Noise sensativity score with model {genai_model} and {embedding}: {result.value}"
    )


if __name__ == "__main__":
    asyncio.run(main())
