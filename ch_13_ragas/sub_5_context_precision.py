import asyncio
import os
import sys
from unittest.mock import MagicMock

# Intercept the dead import BEFORE any ragas imports run:
sys.modules["langchain_community.chat_models.vertexai"] = MagicMock()

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecision

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

scorer = ContextPrecision(llm=llm)


async def main():

    # logger.info(
    #     "# Example 1 : Relevant context is first but two irrelevant chunks follow"
    # )

    # result = await scorer.ascore(
    #     user_input="What is photosynthesis?",
    #     reference="Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce glucose and oxygen",
    #     retrieved_contexts=[
    #         "Photosynthesis is the biological process through which plants convert sunlight and carbon dioxide into glucose and oxygen.",
    #         "Plants require water and minerals from the soil for growth.",
    #         "The Amazon rainforest is home to thousands of plant species.",
    #     ],
    # )

    # # logger.info(f"Context Precision score with model {genai_model}: {result.value}")

    # logger.info(
    #     "# Example 2 : All retrieved contexts are relevant and support the reference answer."
    # )

    # result = await scorer.ascore(
    #     user_input="Where is the Eiffel Tower located?",
    #     reference="The Eiffel Tower is located in Paris, France.",
    #     retrieved_contexts=[
    #         "The Eiffel Tower is located on the Champ de Mars in Paris, France.",
    #         "The Eiffel Tower was built in 1889 and stands 330 meters tall in Paris.",
    #     ],
    # )

    # logger.info(f"Context Precision score with model {genai_model}: {result.value}")

    logger.info(
        "# Example 3 : Two completely irrelevant chunks are ranked before the one relevant chunk."
    )

    result = await scorer.ascore(
        user_input="What is the boiling point of water?",
        reference="Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level.",
        retrieved_contexts=[
            "The capital of France is Paris, a major European city known for the Eiffel Tower.",
            "The density of water is 1 gram per cubic centimeter at 4 degrees Celsius.",
            "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure.",
        ],
    )

    logger.info(f"Context Precision score with model {genai_model}: {result.value}")


if __name__ == "__main__":
    asyncio.run(main=main())
