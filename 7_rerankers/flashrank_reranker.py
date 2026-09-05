from pathlib import Path

from dotenv import load_dotenv
from flashrank import Ranker
from langchain_chroma.vectorstores import Chroma
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_community.document_compressors import FlashrankRerank
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console

# shared cache for all projects, so we won't create / download hugging face reranker model everytime
shared_cache = Path.home() / ".flashrank_cache"  # c://users/vicky/.flashrank_cache

# Instantiate the underlying FlashRank Ranker with cache_dir
ranker_client = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=shared_cache)

load_dotenv()
console = Console()

# colors
primary_col = "yellow"
secondary_col = "cyan"

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

# Define the 30 source documents spanning ML, Generative AI, and Cloud
docs = [
    # Machine Learning (10)
    Document(
        page_content="Transformer models rely on self-attention mechanisms to weigh the importance of different tokens in a sequence. Unlike RNNs, transformers process all tokens in parallel, making them highly efficient on modern GPU hardware. This architecture underpins nearly every state-of-the-art NLP model today."
    ),
    Document(
        page_content="Gradient descent is the core optimization algorithm used to train neural networks by iteratively adjusting weights in the direction that minimizes the loss function. Variants such as SGD, Adam, and RMSProp differ in how they accumulate and apply gradient updates. Choosing the right optimizer and learning rate schedule is critical for convergence."
    ),
    Document(
        page_content="Convolutional Neural Networks (CNNs) extract spatial hierarchies from images using learned filter banks applied via convolution operations. Pooling layers reduce spatial dimensions while preserving dominant features. CNNs remain the standard architecture for image classification, object detection, and segmentation tasks."
    ),
    Document(
        page_content="Attention mechanisms allow a model to focus on specific parts of the input when producing each output token. Scaled dot-product attention computes compatibility scores between query and key vectors, then uses them to form a weighted sum of value vectors. Multi-head attention runs this process in parallel across multiple representation subspaces."
    ),
    Document(
        page_content="Regularization techniques such as L1 and L2 weight penalties prevent neural networks from overfitting by discouraging large weight magnitudes. Dropout randomly zeroes activations during training, forcing the network to learn redundant representations. Early stopping monitors validation loss and halts training when generalization stops improving."
    ),
    Document(
        page_content="Batch normalization normalizes the inputs to each layer across the mini-batch dimension, reducing internal covariate shift and allowing higher learning rates. It introduces learnable scale and shift parameters so the network can undo normalization if needed. Layer normalization is preferred in transformer architectures because it normalizes across the feature dimension instead."
    ),
    Document(
        page_content="Recurrent Neural Networks (RNNs) maintain a hidden state updated at each time step, making them naturally suited for sequential data such as text and time series. Long Short-Term Memory (LSTM) units add gating mechanisms to control information flow, alleviating the vanishing gradient problem. Gated Recurrent Units (GRUs) offer a simplified gating structure with comparable performance."
    ),
    Document(
        page_content="Support Vector Machines (SVMs) find the maximum-margin hyperplane that separates classes in feature space. The kernel trick implicitly maps inputs into high-dimensional spaces where linear separation is possible without computing the mapping explicitly. SVMs are effective in high-dimensional settings and remain competitive for tabular classification tasks."
    ),
    Document(
        page_content="Decision trees partition the feature space using axis-aligned splits chosen to maximize information gain or minimize Gini impurity. Ensemble methods like Random Forests aggregate many trees trained on bootstrap samples, while Gradient Boosted Trees train sequentially, each tree correcting the residuals of the previous one. These methods dominate structured data competitions."
    ),
    Document(
        page_content="Reinforcement learning trains an agent to maximize cumulative reward by interacting with an environment. The agent learns a policy that maps observations to actions, guided by the Bellman equation and temporal-difference learning. Proximal Policy Optimization (PPO) and Soft Actor-Critic (SAC) are popular algorithms for continuous control tasks."
    ),
    # Generative AI (10)
    Document(
        page_content="Large Language Models (LLMs) are autoregressive transformer models trained on massive text corpora to predict the next token. They acquire broad world knowledge and emergent capabilities such as in-context learning and chain-of-thought reasoning at scale. Models like GPT-4, Claude, and Gemini are deployed across a wide range of language tasks."
    ),
    Document(
        page_content="Retrieval-Augmented Generation (RAG) enhances LLM responses by fetching relevant documents from an external knowledge base before generation. A retriever scores candidate passages against the query and passes the top results as context to the language model. This approach reduces hallucination and allows the model to answer questions about up-to-date or proprietary information."
    ),
    Document(
        page_content="Prompt engineering is the practice of crafting input text to elicit desired behavior from a language model without changing its weights. Techniques include few-shot examples, role assignment, chain-of-thought instructions, and output format specifications. Effective prompts can significantly improve accuracy, consistency, and safety of model outputs."
    ),
    Document(
        page_content="Fine-tuning adapts a pretrained language model to a specific task or domain by continuing training on a labeled dataset. Parameter-efficient methods such as LoRA and QLoRA inject small trainable adapter matrices, reducing the number of updated parameters by orders of magnitude. Instruction tuning on diverse task demonstrations improves zero-shot generalization."
    ),
    Document(
        page_content="Reinforcement Learning from Human Feedback (RLHF) aligns language model outputs with human preferences. A reward model is trained on human comparisons of model outputs, and the LLM policy is then optimized against this reward using PPO. This process is central to making models helpful, harmless, and honest."
    ),
    Document(
        page_content="Diffusion models generate images and other data by learning to reverse a gradual noising process. During training, Gaussian noise is progressively added to data, and the model learns to denoise each step. At inference time, the model starts from pure noise and iteratively refines it into a coherent sample, guided optionally by text prompts."
    ),
    Document(
        page_content="Text embeddings map variable-length text into fixed-size dense vectors in a semantic space where similar meanings cluster together. Models like text-embedding-3-small produce embeddings optimized for retrieval tasks through contrastive training on large datasets. These vectors are stored in vector databases and searched using approximate nearest-neighbor algorithms such as HNSW."
    ),
    Document(
        page_content="Hallucination in LLMs refers to the generation of factually incorrect or fabricated information presented confidently. It arises partly because autoregressive training optimizes for fluency rather than factual accuracy. Mitigation strategies include retrieval augmentation, fact-checking pipelines, and training with factuality-aware reward signals."
    ),
    Document(
        page_content="In-context learning allows LLMs to perform new tasks by including a few demonstrations directly in the prompt, without gradient updates. The model infers the task structure and output format from the examples at inference time. Performance is sensitive to example selection, ordering, and the exact phrasing of instructions."
    ),
    Document(
        page_content="Model alignment ensures that AI systems pursue goals intended by their designers and remain safe under distribution shift. Constitutional AI trains models using a set of principles and self-critique, reducing reliance on human-labeled comparisons. Red-teaming, evaluation suites, and interpretability research are complementary tools in the alignment toolkit."
    ),
    # Cloud (10)
    Document(
        page_content="AWS Lambda is a serverless compute service that runs code in response to events without provisioning or managing servers. Functions are billed per invocation and execution duration, making the cost model attractive for intermittent workloads. Lambda integrates natively with S3, DynamoDB, API Gateway, and over 200 other AWS services."
    ),
    Document(
        page_content="Google Cloud Vertex AI provides a unified platform for training, evaluating, and deploying machine learning models at scale. It supports custom training jobs using any framework, as well as managed pipelines via Vertex Pipelines. Model Registry and Endpoint management simplify the transition from experimentation to production inference."
    ),
    Document(
        page_content="Azure Blob Storage is Microsoft's object storage service for unstructured data such as documents, images, and backups. Data is organized into containers and accessed via REST APIs or client SDKs. Tiered storage classes (Hot, Cool, Archive) allow cost optimization based on access frequency."
    ),
    Document(
        page_content="Kubernetes orchestrates containerized workloads by scheduling pods onto nodes in a cluster and maintaining desired state. Deployments, Services, and Ingress controllers abstract compute, networking, and load balancing. Horizontal Pod Autoscaler adjusts replica counts based on CPU utilization or custom metrics to handle traffic spikes."
    ),
    Document(
        page_content="Serverless computing abstracts infrastructure management by automatically allocating resources on demand and scaling to zero when idle. Providers like AWS, Google Cloud, and Azure offer Function-as-a-Service products with per-invocation pricing. Event-driven architectures built on serverless reduce operational overhead but introduce cold-start latency and execution time limits."
    ),
    Document(
        page_content="Content Delivery Networks (CDNs) cache static assets at edge locations geographically close to end users, reducing latency and origin server load. Services such as CloudFront, Fastly, and Akamai route requests to the nearest point of presence using Anycast routing. Cache invalidation policies and TTL settings must be carefully tuned to balance freshness and performance."
    ),
    Document(
        page_content="Identity and Access Management (IAM) controls which principals can perform which actions on which cloud resources. Policies attach to users, groups, or service accounts and are evaluated at request time against the resource's resource-based policy. Least-privilege design and regular access reviews are fundamental cloud security practices."
    ),
    Document(
        page_content="Cloud auto-scaling monitors resource utilization metrics and automatically adjusts the number of compute instances to match current demand. Target tracking policies maintain a chosen metric value, while step scaling policies define discrete adjustment steps. Auto-scaling groups in AWS, managed instance groups in GCP, and scale sets in Azure implement this pattern."
    ),
    Document(
        page_content="Managed database services such as Amazon RDS, Cloud SQL, and Azure Database for PostgreSQL handle provisioning, patching, backups, and failover automatically. Multi-AZ deployments replicate data synchronously to a standby instance, enabling automatic failover with minimal downtime. Read replicas offload reporting queries from the primary instance to improve throughput."
    ),
    Document(
        page_content="Multi-cloud strategies distribute workloads across two or more cloud providers to avoid vendor lock-in and improve resilience. Abstraction layers such as Terraform, Pulumi, and Crossplane allow infrastructure to be defined in a provider-agnostic way. Data egress costs and network latency between clouds are key operational considerations."
    ),
]

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splits = splitter.split_documents(documents=docs)

vector_store = Chroma.from_documents(
    documents=splits, embedding=embeddings, collection_name="flashrank_reranker_demo"
)

base_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# retrieve with base retrievers and print indexed results
q1 = "How do large language models handle factual errors in their output?"

console.print(f"\nQuery 1: {q1}\n", style="blue")

base_result1 = base_retriever.invoke(input=q1)

for i, doc in enumerate(iterable=base_result1, start=1):
    console.print(f"{i}: {doc.page_content}", style="cyan")

compressor = FlashrankRerank(client=ranker_client, top_n=3)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=base_retriever
)

reranked_results = compression_retriever.invoke(input=q1)

for i, doc in enumerate(reranked_results):
    console.print(f"{i}: {doc.page_content}", style="cyan")

# INFO:flashrank.Ranker:Downloading ms-marco-MiniLM-L-12-v2...
# ms-marco-MiniLM-L-12-v2.zip: 100%|███████████████████████████████████████████████| 21.6M/21.6M [00:04<00:00, 5.27MiB/s]
# INFO:httpx:HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:batchEmbedContents "HTTP/1.1 200 OK"

# Query 1: How do large language models handle factual errors in their output?

# INFO:httpx:HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:batchEmbedContents "HTTP/1.1 200 OK"
# 1: Hallucination in LLMs refers to the generation of factually incorrect or fabricated information presented
# confidently. It arises partly because autoregressive training optimizes for fluency rather than factual accuracy.
# Mitigation strategies include retrieval augmentation, fact-checking pipelines, and training with factuality-aware reward
# signals.
# 2: Large Language Models (LLMs) are autoregressive transformer models trained on massive text corpora to predict the
# next token. They acquire broad world knowledge and emergent capabilities such as in-context learning and
# chain-of-thought reasoning at scale. Models like GPT-4, Claude, and Gemini are deployed across a wide range of language
# tasks.
# 3: Retrieval-Augmented Generation (RAG) enhances LLM responses by fetching relevant documents from an external knowledge
# base before generation. A retriever scores candidate passages against the query and passes the top results as context to
# the language model. This approach reduces hallucination and allows the model to answer questions about up-to-date or
# proprietary information.
# 4: Model alignment ensures that AI systems pursue goals intended by their designers and remain safe under distribution
# shift. Constitutional AI trains models using a set of principles and self-critique, reducing reliance on human-labeled
# comparisons. Red-teaming, evaluation suites, and interpretability research are complementary tools in the alignment
# toolkit.
# 5: Reinforcement Learning from Human Feedback (RLHF) aligns language model outputs with human preferences. A reward
# model is trained on human comparisons of model outputs, and the LLM policy is then optimized against this reward using
# PPO. This process is central to making models helpful, harmless, and honest.
# INFO:httpx:HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:batchEmbedContents "HTTP/1.1 200 OK"
# 0: Large Language Models (LLMs) are autoregressive transformer models trained on massive text corpora to predict the
# next token. They acquire broad world knowledge and emergent capabilities such as in-context learning and
# chain-of-thought reasoning at scale. Models like GPT-4, Claude, and Gemini are deployed across a wide range of language
# tasks.
# 1: Reinforcement Learning from Human Feedback (RLHF) aligns language model outputs with human preferences. A reward
# model is trained on human comparisons of model outputs, and the LLM policy is then optimized against this reward using
# PPO. This process is central to making models helpful, harmless, and honest.
# 2: Retrieval-Augmented Generation (RAG) enhances LLM responses by fetching relevant documents from an external knowledge
# base before generation. A retriever scores candidate passages against the query and passes the top results as context to
# the language model. This approach reduces hallucination and allows the model to answer questions about up-to-date or
# proprietary information.
