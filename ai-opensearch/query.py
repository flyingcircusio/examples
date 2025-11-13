# /// script
# dependencies = [
#   "opensearch-py",
# ]
# ///
#
import json
import os
import sys

from opensearchpy import OpenSearch

with open("config.json") as f:
    config = json.load(f)


AI_KEY = os.environ["FCIO_AI_ACCESS_KEY"]
OPENSEARCH_HOST = "127.0.0.1"
OPENSEARCH_PORT = 9200
INDEX_NAME = "documents"
MODEL_ID = config["opensearch"]["model_id"]

client = OpenSearch(
    hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
    http_compress=True,  # enables gzip compression for request bodies
    use_ssl=False,
)


ACCESS = sys.argv[1]
QUERY_TEXT = sys.argv[2]

results = client.search(
    index=INDEX_NAME,
    body={
        "size": 5,
        "_source": {"excludes": ["passage_embedding"]},
        "query": {
            "neural": {
                "passage_embedding": {
                    "query_text": QUERY_TEXT,
                    "model_id": MODEL_ID,
                    "min_score": 0.7,
                    "filter": {"term": {"access": ACCESS}},
                }
            }
        },
    },
)


for hit in results["hits"]["hits"]:
    print(f"{hit['_score']} {hit['_source']['body']}")
