# /// script
# dependencies = [
#   "opensearch-py",
# ]
# ///

import json
import os
import time

from opensearchpy import OpenSearch

with open("config.json") as f:
    config = json.load(f)

AI_KEY = os.environ["FCIO_AI_ACCESS_KEY"]
AI_ENDPOINT = "ai.whq.fcio.net"
OPENSEARCH_HOST = "127.0.0.1"
OPENSEARCH_PORT = 9200
INDEX_NAME = "documents"


os = OpenSearch(
    hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
    http_compress=True,  # enables gzip compression for request bodies
    use_ssl=False,
)


POST_PROCESS = """
    def json = "{\\"name\\": \\"sentence_embedding\\", \
         \\"data_type\\": \\"FLOAT32\\", \
         \\"shape\\": [" + params.data[0].embedding.length + "], \
         \\"data\\": " + params.data[0].embedding + "}";
    return json;
"""

print("Configuring cluster ...")
os.cluster.put_settings(
    body={
        "persistent": {
            "plugins.ml_commons.only_run_on_ml_node": "false",
            "plugins.ml_commons.native_memory_threshold": "99",
            "plugins.ml_commons.trusted_connector_endpoints_regex": [
                "^https://ai\\.whq\\.fcio\\.net/.*$",
            ],
        }
    }
)

# Model Group
print("Registering Model Group")
response = os.plugins.ml.register_model_group(
    body={
        "name": "remote_fc_ai",
        "description": "A model group for external models hosted by FC",
    }
)
model_group = response["model_group_id"]
print("Model group", model_group)

response = os.plugins.ml.create_connector(
    body={
        "name": "embeddinggemma:300m connector",
        "description": "Connect to FC embeddinggemma:300m",
        "version": 1,
        "protocol": "http",
        "parameters": {
            "endpoint": AI_ENDPOINT,
            "model": "embeddinggemma:300m",
        },
        "credential": {
            "openAI_key": AI_KEY,
        },
        "actions": [
            {
                "action_type": "predict",
                "method": "POST",
                "url": "https://${parameters.endpoint}/openai/v1/embeddings",
                "headers": {"Authorization": "Bearer ${credential.openAI_key}"},
                "request_body": '{ "model": "${parameters.model}", "input": ${parameters.input} }',
                "post_process_function": POST_PROCESS,
            }
        ],
    }
)
connector_id = response["connector_id"]
print("Connector id", connector_id)

response = os.plugins.ml.register_model(
    body={
        "name": "embeddinggemma:300m",
        "function_name": "remote",
        "model_group_id": model_group,
        "description": "Embedding Model",
        "connector_id": connector_id,
    },
)

assert response["status"] == "CREATED"
task_id = response["task_id"]

state = None
while True:
    response = os.plugins.ml.get_task(task_id=task_id)
    state = response["state"]
    print(f"Waiting for model import, current state: {state}")
    if state == "COMPLETED":
        break
    time.sleep(1)
model_id = response["model_id"]
config["opensearch"]["model_id"] = model_id

response = os.ingest.put_pipeline(
    id="document-ingest",
    body={
        "description": "Pipeline to ingest documents",
        "processors": [
            {
                "set": {
                    "description": "Create a new field with the Gemma prompt format",
                    "field": "gemma_prompt",
                    "value": "task: sentence similarity | query: {{{body}}}",
                }
            },
            {
                "text_embedding": {
                    "model_id": model_id,
                    "field_map": {"gemma_prompt": "passage_embedding"},
                }
            },
        ],
    },
)
print(response)

response = os.indices.create(
    index=INDEX_NAME,
    body={
        "settings": {
            "index.knn": True,
            "index.number_of_shards": 2,
            "default_pipeline": "document-ingest",
        },
        "mappings": {
            "properties": {
                "docid": {"type": "keyword"},
                "passage_embedding": {
                    "type": "knn_vector",
                    "dimension": 768,  # Must match the ML output vector
                    "space_type": "cosinesimil",
                },
                "body": {"type": "text"},
                "access": {"type": "keyword"},
                # Add more fields as required
            }
        },
    },
)

print(response)

with open("config.json", "w") as f:
    json.dump(config, f)
