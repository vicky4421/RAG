from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.retrievers import SelfQueryRetriever
from langchain_community.query_constructors.chroma import ChromaTranslator
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from rich.console import Console

load_dotenv()
console = Console()

# colors
primary_col = "yellow"
secondary_col = "cyan"

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
llm = init_chat_model(model="google_genai:gemini-3.1-flash-lite", temperature=0)

# Movies dataset — rich, structured metadata makes self-query filtering meaningful
docs = [
    Document(
        page_content="A masked vigilante fights crime in a corrupt city with the help of a billionaire's technology. An iconic supervillain pushes him to his limits in a battle for Gotham's soul.",
        metadata={
            "title": "The Dark Knight",
            "genre": "action",
            "year": 2008,
            "rating": 9.0,
            "director": "Christopher Nolan",
        },
    ),
    Document(
        page_content="A thief who steals secrets through dream-sharing technology is offered a chance to have his past erased if he can plant an idea in someone's mind. A visually stunning exploration of the subconscious.",
        metadata={
            "title": "Inception",
            "genre": "sci-fi",
            "year": 2010,
            "rating": 8.8,
            "director": "Christopher Nolan",
        },
    ),
    Document(
        page_content="A team of explorers travels through a wormhole in space to find a new habitable planet for humanity. Stunning visuals of black holes and time dilation challenge our understanding of physics.",
        metadata={
            "title": "Interstellar",
            "genre": "sci-fi",
            "year": 2014,
            "rating": 8.6,
            "director": "Christopher Nolan",
        },
    ),
    Document(
        page_content="A programmer discovers that reality is a simulation and joins a rebellion against the machines controlling humanity. A groundbreaking blend of philosophy, martial arts, and bullet-time action.",
        metadata={
            "title": "The Matrix",
            "genre": "sci-fi",
            "year": 1999,
            "rating": 8.7,
            "director": "Lana Wachowski",
        },
    ),
    Document(
        page_content="Two criminals and a mob boss's wife are caught in a web of violence and dark humor over a single eventful day in Los Angeles. Interweaving storylines told out of chronological order.",
        metadata={
            "title": "Pulp Fiction",
            "genre": "drama",
            "year": 1994,
            "rating": 8.9,
            "director": "Quentin Tarantino",
        },
    ),
    Document(
        page_content="A maverick surgeon navigates the chaotic social landscape of a mobile army unit during the Korean War. Sharp satirical comedy disguised as a war film, later adapted into a beloved TV series.",
        metadata={
            "title": "MASH",
            "genre": "comedy",
            "year": 1970,
            "rating": 7.4,
            "director": "Robert Altman",
        },
    ),
    Document(
        page_content="Humanity sends a last-ditch mission to reignite the dying sun with a massive stellar bomb. An intense psychological thriller set in the terrifying emptiness of deep space.",
        metadata={
            "title": "Sunshine",
            "genre": "sci-fi",
            "year": 2007,
            "rating": 7.3,
            "director": "Danny Boyle",
        },
    ),
    Document(
        page_content="A soldier wakes up in another man's body aboard a commuter train just minutes before it explodes, reliving the event repeatedly to identify the bomber. A clever sci-fi thriller about time loops and identity.",
        metadata={
            "title": "Source Code",
            "genre": "sci-fi",
            "year": 2011,
            "rating": 7.5,
            "director": "Duncan Jones",
        },
    ),
]

console.print(f"Created {len(docs)} movie documents", style="blue")

# create vector store
vector_store = Chroma.from_documents(
    documents=docs, embedding=embeddings, collection_name="movies_collection"
)

# AttributeInfo, for structured output
metadata_field_info = [
    AttributeInfo(name="title", description="The title of the movie", type="string"),
    AttributeInfo(
        name="genre",
        description="The genre of the movie (action, sci-fi, drama, comedy)",
        type="string",
    ),
    AttributeInfo(
        name="year", description="The year movie was released", type="integer"
    ),
    AttributeInfo(
        name="rating", description="The IMDb rating of the movie (0-10)", type="float"
    ),
    AttributeInfo(
        name="director", description="The director of the movie", type="string"
    ),
]

# description about page content for symantic meaning
document_content_description = "Brief plot descriptions of movies"

# retriever
retriver = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vector_store,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    structured_query_translator=ChromaTranslator(),
)

# retriver without self query
vanilla_retriever = vector_store.as_retriever(
    search_type="similarity", search_kwargs={"k": 3}
)

q = "What are some sci-fi movies released on and after 2005"

console.print(f"\nQuery: {q}", style="blue")

vanilla_results = vanilla_retriever.invoke(input=q)

console.print(f"Retrieved {len(vanilla_results)} for vanilla retrieval", style="yellow")

for i, doc in enumerate(iterable=vanilla_results, start=1):
    console.print(
        f"{i}: {doc.metadata['title']} | {doc.metadata['year']}", style=primary_col
    )
    console.print(doc.page_content + "\n", style=secondary_col)

# retrieval for self query
self_results = retriver.invoke(input=q)

console.print(f"Retrieved {len(self_results)} for self query retrieval", style="yellow")
for i, doc in enumerate(iterable=self_results, start=1):
    console.print(
        f"{i}: {doc.metadata['title']} | {doc.metadata['year']}", style=primary_col
    )
    console.print(doc.page_content + "\n", style=secondary_col)


# Created 8 movie documents

# Query: What are some sci-fi movies released on and after 2005
# Retrieved 3 for vanilla retrieval
# 1: Source Code | 2011
# A soldier wakes up in another man's body aboard a commuter train just minutes before it explodes, reliving the event
# repeatedly to identify the bomber. A clever sci-fi thriller about time loops and identity.

# 2: Inception | 2010
# A thief who steals secrets through dream-sharing technology is offered a chance to have his past erased if he can plant
# an idea in someone's mind. A visually stunning exploration of the subconscious.

# 3: Interstellar | 2014
# A team of explorers travels through a wormhole in space to find a new habitable planet for humanity. Stunning visuals of
# black holes and time dilation challenge our understanding of physics.

# Direct use of automatic function calling (AFC) in Models.generate_content is not recommended. Instead, we recommend to use AFC in Chat.send_message. Similarly, direct use of AFC in Models.generate_content_stream is not recommended. Instead, we recommend to use AFC in Chat.send_message_stream.
# Retrieved 4 for self query retrieval
# 1: Interstellar | 2014
# A team of explorers travels through a wormhole in space to find a new habitable planet for humanity. Stunning visuals of
# black holes and time dilation challenge our understanding of physics.

# 2: Inception | 2010
# A thief who steals secrets through dream-sharing technology is offered a chance to have his past erased if he can plant
# an idea in someone's mind. A visually stunning exploration of the subconscious.

# 3: Sunshine | 2007
# Humanity sends a last-ditch mission to reignite the dying sun with a massive stellar bomb. An intense psychological
# thriller set in the terrifying emptiness of deep space.

# 4: Source Code | 2011
# A soldier wakes up in another man's body aboard a commuter train just minutes before it explodes, reliving the event
# repeatedly to identify the bomber. A clever sci-fi thriller about time loops and identity.

# NOTE: vanilla retrieval fetched 3 results whereas self-query retrieval fetched 4 results
