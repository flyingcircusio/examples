# /// script
# dependencies = [
#   "opensearch-py",
# ]
# ///
#
import os
import pprint
import sys

from opensearchpy import OpenSearch

AI_KEY = os.environ["FCIO_AI_ACCESS_KEY"]
OPENSEARCH_HOST = "127.0.0.1"
OPENSEARCH_PORT = 9200
INDEX_NAME = "documents"
MODEL_ID = "sgoeKpoBIQvP5Ft2CIz3"  # as output by prepare-index

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
            "bool": {
                "filter": [
                    {"term": {"access": ACCESS}},
                ],
                "must": [
                    {
                        "neural": {
                            "passage_embedding": {
                                "query_text": QUERY_TEXT,
                                "model_id": MODEL_ID,
                                "k": 2,
                            }
                        }
                    },
                ],
            }
        },
    },
)


pprint.pprint(results)
