# /// script
# dependencies = [
#   "opensearch-py",
# ]
# ///
#
import json
import os
from pathlib import Path

from opensearchpy import OpenSearch

AI_KEY = os.environ["FCIO_AI_ACCESS_KEY"]
OPENSEARCH_HOST = "127.0.0.1"
OPENSEARCH_PORT = 9200
INDEX_NAME = "documents"

client = OpenSearch(
    hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
    http_compress=True,  # enables gzip compression for request bodies
    use_ssl=False,
)

for f in Path("documents").glob("*.json"):
    print(f.stem)
    document = json.load(f.open())
    response = client.index(
        index=INDEX_NAME, body=document, id=f.stem, refresh=True
    )
