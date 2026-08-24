from pathlib import Path
from pprint import pp  # prettyprint like print() but with structured format

from langchain_community.document_loaders import TextLoader

# define path for the text file
file_path = Path("./knowledge_source/transformers.txt")

# define the loader
loader = TextLoader(file_path=file_path)

# load the docs
documents = loader.load()

# pp(documents)

# [Document(metadata={'source': 'knowledge_source\\transformers.txt'}, page_content='# Transformer Model in Large Language Models (LLMs)\n\nThis note explains the Transformer model as used in modern Large Language Models (LLMs). It uses simple language and clear structure. You’ll get the key ideas, components, how training and inference work, strengths and limits, and common improvements.\n\n---\n\n## 1. Big picture - Why Transformers matter\n- Transformers are the core architecture behind most modern LLMs (GPT, BERT, PaLM, LLaMA).\n- They replaced recurrent and convolutional models because they handle long-range context efficiently and are highly parallelizable on GPUs/TPUs.\n- The main innovation: self-attention, which lets the model weigh relationships among all tokens in a sequence.\n\n---\n\n## 2. Core building blocks\nA Transformer layer consists of a few repeating parts:\n\n1. Multi-Head Self-Attention\n   - Computes attention scores between every pair of tokens.\n   - Produces a weighted sum of token representations so each token can "look at" others.\n   - Multi-head: runs several attention computations in parallel, each with different learned projections to capture different types of relationships.\n\n2. Feed-Forward Network (FFN)\n   - A small neural network applied independently to each position.\n   - Typically two linear ...

pp(documents[0].metadata)

# {'source': 'knowledge_source\\transformers.txt'}
