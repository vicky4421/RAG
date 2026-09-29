import asyncio
import os
import sys
from unittest.mock import MagicMock

# Intercept the dead import BEFORE any ragas imports run:
sys.modules["langchain_community.chat_models.vertexai"] = MagicMock()

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import NoiseSensitivity

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

# 3. Pass to metric
scorer = NoiseSensitivity(llm=llm)


async def main():
    # logger.info(
    #     "# Example 1: Response is mostly correct but 'contributes to financial stability' is a claim drawn from the noisy context rather than the core reference."
    # )

    # result = await scorer.ascore(
    #     user_input="What is the Life Insurance Corporation of India (LIC) known for?",
    #     response="The Life Insurance Corporation of India (LIC) is the largest insurance company in India, known for its vast portfolio of investments. LIC contributes to the financial stability of the country.",
    #     reference="The Life Insurance Corporation of India (LIC) is the largest insurance company in India, established in 1956 through the nationalization of the insurance industry. It is known for managing a large portfolio of investments",
    #     retrieved_contexts=[
    #         "The Life Insurance Corporation of India (LIC) was established in 1956 following the nationalization of the insurance industry in India.",
    #         "LIC is the largest insurance company in India, with a vast network of policyholders and huge investments.",
    #         "As the largest institutional investor in India, LIC manages substantial funds, contributing to the financial stability of the country.",
    #         "The Indian economy is one of the fastest-growing major economies in the world, thanks to sectors like finance, technology, and manufacturing.",
    #     ],
    # )

    logger.info(
        "# Example 2 : Response stays grounded in the reference despite noisy contexts. The Louvre and French culture contexts are completely ignored by the response"
    )

    result = await scorer.ascore(
        user_input="Where is the Eiffel Tower located?",
        response="The Eiffel Tower is located in Paris, France.",
        reference="The Eiffel Tower is located in Paris, France.",
        retrieved_contexts=[
            "The Eiffel Tower is a landmark located in Paris, France.",
            "Paris is also home to the Louvre Museum, which contains thousands of artworks including the Mona Lisa.",
            "France is known for its cuisine, wine, and fashion industry.",
        ],
    )

    logger.info(f"Noise sensativity score with model {genai_model}: {result.value}")


if __name__ == "__main__":
    asyncio.run(main())
