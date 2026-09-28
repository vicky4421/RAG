import sys
from unittest.mock import MagicMock

# Intercept the dead import BEFORE any ragas imports run:
sys.modules["langchain_community.chat_models.vertexai"] = MagicMock()

import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRecall

from utils.logger import get_logger

load_dotenv()
logger = get_logger(log_name="ragas.log")

model_3_6 = "gemini-3.6-flash"
model_3_1 = "gemini-3.1-flash-lite"

genai_model = model_3_1

# google_client = genai.Client()
google_client = AsyncOpenAI(
    api_key=os.getenv(key="GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

llm = llm_factory(model=genai_model, client=google_client)

# 3. Pass to metric
scorer = ContextRecall(llm=llm)


async def main():

    # logger.info(
    #     "# Example 1: Contexts cover symptoms but miss the cause (insulin resistance / obesity)"
    # )

    # result = await scorer.ascore(
    #     user_input="What are symptoms and causes of type 2 diabetes?",
    #     retrieved_contexts=[
    #         "Type 2 diabetes symptoms include frequent urination and excessive thirst.",
    #         "People with type 2 diabetes ofter experience fatigue and blurred vision.",
    #     ],
    #     reference="Type 2 diabetes caused by insulin resistance, often linked to obesity and a sedetary lifestyle. Its symptoms include frequent urination, excessive thirst, fatigue, blurred vision, and slow-healing sores.",
    # )

    # logger.info(
    #     "# Example 2 : The retrieved context fully covers every claim in the ground truth"
    # )

    # result = await scorer.ascore(
    #     user_input="Where is eiffel tower located?",
    #     retrieved_contexts=[
    #         "The Eiffel tower is a wrought iron lattice tower located on the Champ De Mars in Paris, France."
    #     ],
    #     reference="The Eiffel tower is located in France.",
    # )

    # logger.info(f"Context Recall score with model {genai_model}: {result.value}")

    logger.info(
        "# Example 3: Context only addresses treatment; ground truth covers causes and symptoms"
    )

    result = await scorer.ascore(
        user_input="What causes and characterizes Parkinson's disease?",
        retrieved_contexts=[
            "Parkinson's disease is managed using medications such as levodopa and physical therapy to improve quality of life."
        ],
        reference="Parkinson's disease is caused by the loss of dopamine-producing neurons in the brain. It is characterized by tremors, stiffness, slowness of movement, and balance problems.",
    )

    logger.info(f"Context Recall score with model {genai_model}: {result.value}")


if __name__ == "__main__":
    asyncio.run(main())
