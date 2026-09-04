from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console

load_dotenv()

console = Console()

# colors
primary_col = "yellow"
secondary_col = "cyan"

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
llm = init_chat_model(model="google_genai:gemini-3.1-flash-lite", temperature=0)

docs = [
    Document(
        page_content=(
            "Biotechnology companies are developing novel protein-based therapies that target specific "
            "disease pathways with unprecedented precision. Synthetic biology techniques allow scientists "
            "to engineer microorganisms that produce pharmaceutical compounds at industrial scale. "
            "Bioreactor technologies have dramatically reduced the cost of producing monoclonal antibodies, "
            "making treatments for autoimmune diseases and cancers more accessible. Microbiome research is "
            "revealing how manipulating gut bacteria can influence everything from mental health to "
            "metabolic disorders."
        ),
        metadata={"topic": "biotechnology"},
    ),
    Document(
        page_content=(
            "Zero-trust architecture has become the gold standard for enterprise network security, "
            "requiring continuous verification rather than relying on perimeter defenses. Machine learning "
            "models now detect anomalous network behavior in real time, reducing the window between "
            "intrusion and detection from months to minutes. Ransomware attacks on critical infrastructure "
            "have forced governments to establish mandatory incident reporting requirements for healthcare "
            "and energy sectors. Post-quantum cryptography standards are being finalized to protect "
            "sensitive data against future quantum computing threats."
        ),
        metadata={"topic": "cybersecurity"},
    ),
    Document(
        page_content=(
            "Brain-computer interfaces are enabling paralyzed patients to control prosthetic limbs and "
            "communicate using only their neural signals. Optogenetics allows researchers to activate or "
            "silence specific neuron populations with light, accelerating the understanding of neural "
            "circuit function and disease. Advanced neuroimaging techniques using fMRI and "
            "magnetoencephalography are mapping brain connectivity with millimeter precision, unlocking "
            "new treatments for depression and PTSD. Neurofeedback therapies are showing promise for "
            "cognitive rehabilitation following traumatic brain injuries."
        ),
        metadata={"topic": "neuroscience"},
    ),
    Document(
        page_content=(
            "Perovskite solar cells have achieved efficiency ratings exceeding 33%, surpassing traditional "
            "silicon panels and promising dramatically lower manufacturing costs. Grid-scale battery "
            "storage using iron-air and sodium-ion technologies is making renewable energy dispatchable "
            "around the clock without relying on rare earth metals. Offshore floating wind farms are "
            "expanding into deep-water regions previously inaccessible to fixed-foundation turbines, "
            "multiplying available wind energy capacity. Green hydrogen produced via electrolysis is "
            "emerging as a critical energy carrier for decarbonizing heavy industry and long-haul "
            "transport."
        ),
        metadata={"topic": "renewable_energy"},
    ),
    Document(
        page_content=(
            "Surgical robots equipped with haptic feedback allow surgeons to perform minimally invasive "
            "procedures with sub-millimeter precision, reducing patient recovery times significantly. "
            "Collaborative robots in manufacturing now work safely alongside humans using advanced "
            "computer vision and force sensing, without the need for physical barriers. Autonomous mobile "
            "robots are transforming warehouse logistics, optimizing pick-and-place operations and "
            "reducing fulfillment errors. Soft robots inspired by biological organisms are being developed "
            "for delicate tasks in agriculture, search-and-rescue, and medical drug delivery."
        ),
        metadata={"topic": "robotics"},
    ),
    Document(
        page_content=(
            "Base editing and prime editing technologies offer more precise alternatives to CRISPR-Cas9, "
            "enabling single-letter corrections to the genome without creating double-strand breaks. "
            "Gene therapy trials using adeno-associated virus vectors have achieved functional cures for "
            "hemophilia B and spinal muscular atrophy. Epigenome editing tools allow researchers to "
            "switch genes on or off without altering the underlying DNA sequence, opening new avenues "
            "for treating complex diseases. Polygenic risk scoring combined with germline analysis is "
            "enabling predictive medicine that identifies disease susceptibility decades before symptoms "
            "appear."
        ),
        metadata={"topic": "genetic_engineering"},
    ),
]

console.print(f"Created {len(docs)} documents", style="blue")

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

docs = splitter.split_documents(documents=docs)

vector_store = InMemoryVectorStore.from_documents(documents=docs, embedding=embeddings)

base_retriver = vector_store.as_retriever(search_kwargs={"k": 3})

retriever = MultiQueryRetriever.from_llm(retriever=base_retriver, llm=llm)

q = "How are modern technologies improving human health?"

console.print(f"Query: {q}", style="cyan3")

results = retriever.invoke(input=q)

console.print(f"Retrieved {len(results)}", style="yellow")
for i, doc in enumerate(iterable=results, start=1):
    console.print(f"{i}: {doc.metadata['topic']}", style=primary_col)
    console.print(doc.page_content + "\n", style=secondary_col)

# Created 6 documents
# Query: How are modern technologies improving human health?
# Direct use of automatic function calling (AFC) in Models.generate_content is not recommended. Instead, we recommend to use AFC in Chat.send_message. Similarly, direct use of AFC in Models.generate_content_stream is not recommended. Instead, we recommend to use AFC in Chat.send_message_stream.
# Retrieved 4
# 1: robotics
# Surgical robots equipped with haptic feedback allow surgeons to perform minimally invasive procedures with
# sub-millimeter precision, reducing patient recovery times significantly. Collaborative robots in manufacturing now work
# safely alongside humans using advanced computer vision and force sensing,

# 2: biotechnology
# at industrial scale. Bioreactor technologies have dramatically reduced the cost of producing monoclonal antibodies,
# making treatments for autoimmune diseases and cancers more accessible. Microbiome research is revealing how manipulating
# gut bacteria can influence everything from mental health to

# 3: neuroscience
# of neural circuit function and disease. Advanced neuroimaging techniques using fMRI and magnetoencephalography are
# mapping brain connectivity with millimeter precision, unlocking new treatments for depression and PTSD. Neurofeedback
# therapies are showing promise for cognitive rehabilitation

# 4: genetic_engineering
# functional cures for hemophilia B and spinal muscular atrophy. Epigenome editing tools allow researchers to switch genes
# on or off without altering the underlying DNA sequence, opening new avenues for treating complex diseases. Polygenic
# risk scoring combined with germline analysis is enabling
