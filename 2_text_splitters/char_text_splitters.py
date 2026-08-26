from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

# sample text
text = """Artificial intelligence is transforming technology and shaping the future. 
Machine learning algorithms are becoming more sophisticated every day.
Deep learning models can now process vast amounts of data efficiently.

Natural language processing has made significant strides in recent years.
Computer vision systems can now identify objects with remarkable accuracy.
Reinforcement learning is enabling robots to learn complex tasks autonomously.

The impact of AI extends across multiple industries including healthcare, finance, and transportation.
Ethical considerations around AI development are becoming increasingly important.
Researchers are working on making AI systems more transparent and explainable."""

# separator: for character "", for word " ", for line "\n", for paragraph "\n\n"
# chunk size: 100 characters
# length_function: function which calculates the size of chunk

splitter = CharacterTextSplitter(
    separator="", chunk_size=100, chunk_overlap=10, length_function=len
)

chunks = splitter.split_text(text=text)

print(chunks)
print(f"No. of chunks: {len(chunks)}")

# Output for character wise chunk
# ['Artificial intelligence is transforming technology and shaping the future. \nMachine learning algorit',
# 'hms are becoming more sophisticated every day.\nDeep learning models can now process vast amounts of',
# 'data efficiently.\n\nNatural language processing has made significant strides in recent years.\nCompute',
# 'r vision systems can now identify objects with remarkable accuracy.\nReinforcement learning is enabli',
# 'ng robots to learn complex tasks autonomously.\n\nThe impact of AI extends across multiple industries',
# 'including healthcare, finance, and transportation.\nEthical considerations around AI development are',
# 'becoming increasingly important.\nResearchers are working on making AI systems more transparent and e',
# 'xplainable.']
# No. of chunks: 8

# Output for word wise chunk
# ['Artificial intelligence is transforming technology and shaping the future. \nMachine learning',
# 'algorithms are becoming more sophisticated every day.\nDeep learning models can now process vast',
# 'amounts of data efficiently.\n\nNatural language processing has made significant strides in recent',
# 'years.\nComputer vision systems can now identify objects with remarkable accuracy.\nReinforcement',
# 'learning is enabling robots to learn complex tasks autonomously.\n\nThe impact of AI extends across',
# 'multiple industries including healthcare, finance, and transportation.\nEthical considerations around',
# 'AI development are becoming increasingly important.\nResearchers are working on making AI systems',
# 'more transparent and explainable.']
# No. of chunks: 8

# Output for line wise chunk
# Created a chunk of size 102, which is longer than the specified 100
# ['Artificial intelligence is transforming technology and shaping the future.',
# 'Machine learning algorithms are becoming more sophisticated every day.',
# 'Deep learning models can now process vast amounts of data efficiently.',
# 'Natural language processing has made significant strides in recent years.',
# 'Computer vision systems can now identify objects with remarkable accuracy.',
# 'Reinforcement learning is enabling robots to learn complex tasks autonomously.',
# 'The impact of AI extends across multiple industries including healthcare, finance, and transportation.',
# 'Ethical considerations around AI development are becoming increasingly important.',
# 'Researchers are working on making AI systems more transparent and explainable.']
# No. of chunks: 9

# Output for paragraph wise chunk
# Created a chunk of size 217, which is longer than the specified 100
# Created a chunk of size 227, which is longer than the specified 100
# ['Artificial intelligence is transforming technology and shaping the future. \nMachine learning algorithms are becoming more sophisticated every day.\nDeep learning models can now process vast amounts of data efficiently.',
#
# 'Natural language processing has made significant strides in recent years.\nComputer vision systems can now identify objects with remarkable accuracy.\nReinforcement learning is enabling robots to learn complex tasks autonomously.',
#
#  'The impact of AI extends across multiple industries including healthcare, finance, and transportation.\nEthical considerations around AI development are becoming increasingly important.\nResearchers are working on making AI systems more transparent and explainable.']
# No. of chunks: 3

# NOTE: If created chunk size is greater than the threshold mentioned then we should increase the threshold otherwise it will hurt its semantic meaning, e.g. if we keep threshold to 250 characters in paragraph wise chunk, it will match the chunk size and threshold and we get the clean data.

# Output for paragraph wise chunk with 250 character threshold
# [
# 'Artificial intelligence is transforming technology and shaping the future. \nMachine learning algorithms are becoming more sophisticated every day.\nDeep learning models can now process vast amounts of data efficiently.',
#
# 'Natural language processing has made significant strides in recent years.\nComputer vision systems can now identify objects with remarkable accuracy.\nReinforcement learning is enabling robots to learn complex tasks autonomously.',
#
# 'The impact of AI extends across multiple industries including healthcare, finance, and transportation.\nEthical considerations around AI development are becoming increasingly important.\nResearchers are working on making AI systems more transparent and explainable.'
# ]
# No. of chunks: 3

# CHUNK OVERLAP
#   When we split the text characterwise, the words get splits and it left no meaning.
#   ...learning algorit', 'hms are...   <--- here algorithm broken in two
#   when we provide the value for chunk overlap arg e.g 10, then we are actually asking the splitter that you can overlap 10 characters to maintain the context.
#   ------------
#          -----------
#          | ^ |
#          Overlap

# Output with characterwise splitting threshold 100 chars and chunk overlap 10
# [
# 'Artificial intelligence is transforming technology and shaping the future. \nMachine learning algorit',
# 'ng algorithms are becoming more sophisticated every day.\nDeep learning models can now process vast a',
# 'ess vast amounts of data efficiently.\n\nNatural language processing has made significant strides in r',
# 'rides in recent years.\nComputer vision systems can now identify objects with remarkable accuracy.\nRe',
# 'curacy.\nReinforcement learning is enabling robots to learn complex tasks autonomously.\n\nThe impact o',
# 'e impact of AI extends across multiple industries including healthcare, finance, and transportation.',
# 'portation.\nEthical considerations around AI development are becoming increasingly important.\nResearc',
# 't.\nResearchers are working on making AI systems more transparent and explainable.'
# ]
# No. of chunks: 8

# We're allowing 10 characters to overlap to maintain the context, now though in first chunk 'algorit' cuts without completing the word but in next chunk 'ng algorithms are becoming', spliltter added 10 characters upfront in next chunk and 'algorithm' gets completed which now maintain the context as well as semantic meaning.

# TOKEN WISE SPLITTING
token_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=50, chunk_overlap=5
)

chunks = token_splitter.split_text(text=text)

print(chunks)
print(f"No. of chunks: {len(chunks)}")

# Output with token wise splitting
# [
# 'Artificial intelligence is transforming technology and shaping the future. \nMachine learning algorithms are becoming more sophisticated every day.\nDeep learning models can now process vast amounts of data efficiently.',
# 'Natural language processing has made significant strides in recent years.\nComputer vision systems can now identify objects with remarkable accuracy.\nReinforcement learning is enabling robots to learn complex tasks autonomously.',
# 'The impact of AI extends across multiple industries including healthcare, finance, and transportation.\nEthical considerations around AI development are becoming increasingly important.\nResearchers are working on making AI systems more transparent and explainable.']
# Chunk length: 3


# DOCUMENT SPLITTING
# mimicking document
docs = [Document(page_content=text, metadata={"source": "Text paragraphs"})]

doc_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10, separator="")

chunks = doc_splitter.split_documents(documents=docs)

print(chunks)
print(f"No. of chunks: {len(chunks)}")

# Output with document splitting
# [
# Document(metadata={'source': 'Text paragraphs'}, page_content='Artificial intelligence is transforming technology and shaping the future. \nMachine learning algorit'),
# Document(metadata={'source': 'Text paragraphs'}, page_content='ng algorithms are becoming more sophisticated every day.\nDeep learning models can now process vast a'),
# Document(metadata={'source': 'Text paragraphs'}, page_content='ess vast amounts of data efficiently.\n\nNatural language processing has made significant strides in r'), Document(metadata={'source': 'Text paragraphs'}, page_content='rides in recent years.\nComputer vision systems can now identify objects with remarkable accuracy.\nRe'), Document(metadata={'source': 'Text paragraphs'}, page_content='curacy.\nReinforcement learning is enabling robots to learn complex tasks autonomously.\n\nThe impact o'), Document(metadata={'source': 'Text paragraphs'}, page_content='e impact of AI extends across multiple industries including healthcare, finance, and transportation.'), Document(metadata={'source': 'Text paragraphs'}, page_content='portation.\nEthical considerations around AI development are becoming increasingly important.\nResearc'), Document(metadata={'source': 'Text paragraphs'}, page_content='t.\nResearchers are working on making AI systems more transparent and explainable.')]
# No. of chunks: 8
