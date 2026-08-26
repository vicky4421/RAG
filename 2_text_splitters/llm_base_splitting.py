from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

load_dotenv()

text = """Artificial intelligence is transforming technology and shaping the future.
Machine learning algorithms are becoming more sophisticated every day.
Deep learning models can now process vast amounts of data efficiently.
Neural networks are inspired by the human brain's structure.
The best pasta recipes include fresh ingredients and proper cooking techniques.
Italian cuisine emphasizes quality olive oil and regional cheeses.
Authentic carbonara uses guanciale, eggs, pecorino romano, and black pepper.
Cooking pasta al dente ensures the best texture and flavor.
Climate change is affecting ecosystems worldwide.
Rising temperatures are causing glaciers to melt at unprecedented rates.
Scientists warn that immediate action is needed to reduce carbon emissions.
Renewable energy sources offer hope for a sustainable future."""


class Chunk(BaseModel):
    chunk_text: str
    summary: str


class Chunker(BaseModel):
    chunks: list[Chunk]


llm = init_chat_model("google_genai:gemini-3.1-flash-lite")

llm_with_structured_output = llm.with_structured_output(schema=Chunker)

prompt = ChatPromptTemplate(
    messages=[
        (
            "system",
            """You are an expert Text Chunker that splits the given text and outputs them as a 
     list of strings. You understand the natural topic boundaries of text and 
     also do not change the existing text. You just split the text wherever applicable.
     Once you create the chunk, you also generate a 1-2 line summary of the chunk also""",
        ),
        ("human", "splite the given text into chunks \n text: {text}"),
    ],
    input_variables=["text"],
)

chain = prompt | llm_with_structured_output

response = chain.invoke({"text": text})

print(response)

# chunks=[Chunk(chunk_text="Artificial intelligence is transforming technology and shaping the future. Machine learning algorithms are becoming more sophisticated every day. Deep learning models can now process vast amounts of data efficiently. Neural networks are inspired by the human brain's structure.",
# summary='This section covers advancements in artificial intelligence, machine learning, deep learning, and neural networks.'),
# Chunk(chunk_text='The best pasta recipes include fresh ingredients and proper cooking techniques. Italian cuisine emphasizes quality olive oil and regional cheeses. Authentic carbonara uses guanciale, eggs, pecorino romano, and black pepper. Cooking pasta al dente ensures the best texture and flavor.',
# summary='This section provides an overview of Italian cooking principles and essential ingredients for authentic pasta dishes.'),
# Chunk(chunk_text='Climate change is affecting ecosystems worldwide. Rising temperatures are causing glaciers to melt at unprecedented rates. Scientists warn that immediate action is needed to reduce carbon emissions. Renewable energy sources offer hope for a sustainable future.',
# summary='This section discusses the global impact of climate change and emphasizes the importance of renewable energy and carbon reduction.')]
# no. of chunks: 3

docs = [
    Document(page_content=chunk.chunk_text, metadata={"summary": chunk.summary})
    for chunk in response.chunks
]

print(docs)

# [
# Document(metadata={'summary': 'This section covers the advancements and core concepts of artificial intelligence and machine learning.'}, page_content="Artificial intelligence is transforming technology and shaping the future. Machine learning algorithms are becoming more sophisticated every day. Deep learning models can now process vast amounts of data efficiently. Neural networks are inspired by the human brain's structure."),
# Document(metadata={'summary': 'This part details the essential ingredients and methods for preparing authentic Italian pasta dishes.'}, page_content='The best pasta recipes include fresh ingredients and proper cooking techniques. Italian cuisine emphasizes quality olive oil and regional cheeses. Authentic carbonara uses guanciale, eggs, pecorino romano, and black pepper. Cooking pasta al dente ensures the best texture and flavor.'),
# Document(metadata={'summary': 'This segment discusses the global impact of climate change and the importance of adopting renewable energy.'}, page_content='Climate change is affecting ecosystems worldwide. Rising temperatures are causing glaciers to melt at unprecedented rates. Scientists warn that immediate action is needed to reduce carbon emissions. Renewable energy sources offer hope for a sustainable future.')
# ]
