from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """Hi how are you
my name is rahul

i am teaching RAG
we're learning about RAG"""

rsplitter = RecursiveCharacterTextSplitter(
    chunk_size=10,
    chunk_overlap=0,
    separators=["\n\n", "\n"],
    keep_separator=False,
)

chunk1 = rsplitter.split_text(text=text)

print(chunk1)
print(f"No. of chunks: {len(chunk1)}")

# Output with default separators, i.e paragraph, sentences, words, characters
# ['Hi how are', 'you', 'my name', 'is rahul', 'i am', 'teaching', 'RAG', "we're", 'learning', 'about RAG']
# No. of chunks: 10

# Output with separators paragraph, sentences
# ['Hi how are you', 'my name is rahul', 'i am teaching RAG', "we're learning about RAG"]
# No. of chunks: 4

example_text = """Artificial intelligence is transforming technology and shaping the future.

Machine learning algorithms are becoming more sophisticated every day. 
Deep learning models can now process vast amounts of data efficiently.

Natural language processing has made significant strides in recent years.
Transformers architecture revolutionized the field in 2017.
Models like GPT and BERT have set new benchmarks.

Computer vision systems can now identify objects with remarkable accuracy.
Convolutional neural networks excel at image recognition tasks.
Self-driving cars rely heavily on advanced computer vision.

The impact of AI extends across multiple industries including healthcare, finance, and transportation.
Ethical considerations around AI development are becoming increasingly important.
Researchers are working on making AI systems more transparent and explainable."""

r2splitter = RecursiveCharacterTextSplitter(
    chunk_size=150, chunk_overlap=20, keep_separator=False
)

chunk2 = r2splitter.split_text(example_text)

print(chunk2)
print(f"No. of chunks: {len(chunk2)}")

# [
# 'Artificial intelligence is transforming technology and shaping the future.',
# 'Machine learning algorithms are becoming more sophisticated every day. \nDeep learning models can now process vast amounts of data efficiently.',
# 'Natural language processing has made significant strides in recent years.\nTransformers architecture revolutionized the field in 2017.',
# 'Models like GPT and BERT have set new benchmarks.', 'Computer vision systems can now identify objects with remarkable accuracy.\nConvolutional neural networks excel at image recognition tasks.',
# 'Self-driving cars rely heavily on advanced computer vision.',
# 'The impact of AI extends across multiple industries including healthcare, finance, and transportation.',
# 'Ethical considerations around AI development are becoming increasingly important.',
# 'Researchers are working on making AI systems more transparent and explainable.'
# ]
# No. of chunks: 9

# DOCUMENT SPLITTING RECURSIVE
text_list = [text, example_text]

docs = [Document(page_content=text) for text in text_list]

chunk3 = r2splitter.split_documents(docs)

print(chunk3)
print(f"No. of chunks: {len(chunk3)}")

# [Document(metadata={}, page_content="Hi how are you\nmy name is rahul\n\ni am teaching RAG\nwe're learning about RAG"), Document(metadata={}, page_content='Artificial intelligence is transforming technology and shaping the future.'), Document(metadata={}, page_content='Machine learning algorithms are becoming more sophisticated every day. \nDeep learning models can now process vast amounts of data efficiently.'), Document(metadata={}, page_content='Natural language processing has made significant strides in recent years.\nTransformers architecture revolutionized the field in 2017.'), Document(metadata={}, page_content='Models like GPT and BERT have set new benchmarks.'), Document(metadata={}, page_content='Computer vision systems can now identify objects with remarkable accuracy.\nConvolutional neural networks excel at image recognition tasks.'), Document(metadata={}, page_content='Self-driving cars rely heavily on advanced computer vision.'), Document(metadata={}, page_content='The impact of AI extends across multiple industries including healthcare, finance, and transportation.'), Document(metadata={}, page_content='Ethical considerations around AI development are becoming increasingly important.'), Document(metadata={}, page_content='Researchers are working on making AI systems more transparent and explainable.')]
# No. of chunks: 10
