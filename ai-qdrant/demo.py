import argparse
import os
import sys

from openai import OpenAI
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct

parser = argparse.ArgumentParser(
    description="Vector search demo with qdrant and Nomic via the Flying Circus OpenAI-compatible API."
)

parser.add_argument(
    "--query",
    default="Who was the best santa?",
    # Who brought presents?
    # Who did not smile?
    help="The search query to perform",
)

args = parser.parse_args()
query_str = args.query

collection_name = "demo_collection"
model_name = "Nomic-embed-text:v1.5"

# Those depend on the model and the task.
index_prompt = "search_document: "
query_prompt = "search_query: "

# Setup

oai = OpenAI(
    base_url=os.environ.get(
        "FCIO_AI_ENDPOINT", "https://ai.rzob.fcio.net/openai/v1"
    ),
    api_key=os.environ.get("FCIO_AI_ACCESS_KEY"),
)

qdrant = QdrantClient(":memory:")
qdrant.create_collection(
    collection_name,
    vectors_config=models.VectorParams(
        size=768,
        distance=models.Distance.COSINE,
    ),
)

# Index
docs = {
    1: {
        "document": "Billy Bob Thornton was the best santa claus because he was so funny and brought presents.",
        "source": "Theuni",
    },
    2: {
        "document": "Batman was the worst santa claus because he never smiles and did not bring any presents.",
        "source": "The Joker",
    },
    3: {
        "document": "Red wine tastes better with chicken than white whine.",
        "source": "Some french person",
    },
    4: {
        "document": "White whine is great at a summer BBQ.",
        "source": "Some american person",
    },
}

for id, doc in docs.items():
    text = doc["document"]
    embed = oai.embeddings.create(
        model=model_name,
        input=index_prompt + text,
        encoding_format="float",
    )
    vector = embed.data[0].embedding
    payload = doc.copy()
    del payload["document"]
    qdrant.upsert(
        collection_name=collection_name,
        wait=True,
        points=[PointStruct(id=id, vector=vector, payload=payload)],
    )

# Search

r = oai.embeddings.create(
    model=model_name,
    input=query_prompt + query_str,
    encoding_format="float",
)
query = r.data[0].embedding

search_result = qdrant.query_points(
    collection_name=collection_name,
    query=query,
    with_payload=True,
    limit=3,
).points

for result in search_result:
    print(result.score)
    print(docs[result.id]["document"])
