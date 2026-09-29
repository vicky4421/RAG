import asyncio
import os
import sys
from unittest.mock import MagicMock

# Intercept the dead import BEFORE any ragas imports run:
sys.modules["langchain_community.chat_models.vertexai"] = MagicMock()

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

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

scorer = Faithfulness(llm=llm)


async def main():

    # logger.info(
    #     "# Example 1: Response is mostly grounded but adds one unsupported claim has been proven to prevent all forms of cancer' is not in the context"
    # )

    # result = await scorer.ascore(
    #     user_input="What are the health benefits of green tea?",
    #     response="Green tea contains antioxidants that help reduce inflammation. It also boosts metabolism and has been proven to prevent all forms of cancer.",
    #     retrieved_contexts=[
    #         "Green tea is rich in antioxidants, particularly catechins, which help reduce inflammation and oxidative stress.",
    #         "Studies suggest green tea may modestly boost metabolic rate.",
    #     ],
    # )

    # logger.info(f"Faithfulness score with model {genai_model}: {result.value}")

    # logger.info(
    #     "# Example 2 : Every claim in the response is directly supported by the context"
    # )

    # result = await scorer.ascore(
    #     user_input="When was the first Super Bowl played?",
    #     response="The first Super Bowl was played on January 15, 1967, at the Los Angeles Memorial Coliseum.",
    #     retrieved_contexts=[
    #         "The First AFL-NFL World Championship Game was played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles, California.",
    #     ],
    # )

    # logger.info(f"Faithfulness score with model {genai_model}: {result.value}")

    logger.info(
        "# Example 3: Response introduces multiple facts not found in the context. Context only states the speed; travel time to Earth and Earth-circling claim are hallucinated."
    )

    result = await scorer.ascore(
        user_input="What is the speed of light?",
        response="The speed of light is approximately 3x10^8 meters per second. It takes light about 8 minutes to travel from the Sun to Earth, and light can circle the Earth 7.5 times in one second.",
        retrieved_contexts=[
            "The speed of light in a vacuum is approximately 299,792,458 meters per second",
        ],
    )

    logger.info(f"Faithfulness score with model {genai_model}: {result.value}")


if __name__ == "__main__":
    asyncio.run(main())
