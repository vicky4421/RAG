from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings

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

sem_splitter = SemanticChunker(
    embeddings=GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview"),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1,
)

chunks = sem_splitter.split_text(text=text)

print(chunks)
print(f"no. of chunks: {len(chunks)}")

# [
# "Artificial intelligence is transforming technology and shaping the future. Machine learning algorithms are becoming more sophisticated every day. Deep learning models can now process vast amounts of data efficiently. Neural networks are inspired by the human brain's structure. The best pasta recipes include fresh ingredients and proper cooking techniques.",
# 'Italian cuisine emphasizes quality olive oil and regional cheeses. Authentic carbonara uses guanciale, eggs, pecorino romano, and black pepper. Cooking pasta al dente ensures the best texture and flavor.',
# 'Climate change is affecting ecosystems worldwide.',
# 'Rising temperatures are causing glaciers to melt at unprecedented rates. Scientists warn that immediate action is needed to reduce carbon emissions. Renewable energy sources offer hope for a sustainable future.']
# no. of chunks: 4
