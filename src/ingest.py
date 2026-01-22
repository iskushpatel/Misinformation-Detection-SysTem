import json
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
import config
def ingest_data():
    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    model = SentenceTransformer(config.MODEL_NAME)
    with open("data/knowledge_base.json", "r") as f:
        data = json.load(f)
    if client.collection_exists(config.COLLECTION_NAME):
        client.delete_collection(config.COLLECTION_NAME)
        
    client.create_collection(
        collection_name=config.COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print(f"Created collection: {config.COLLECTION_NAME}")
    points = []
    for entry in data:
        vector = model.encode(entry["text"]).tolist()
        points.append(PointStruct(
            id=entry["id"],
            vector=vector,
            payload={
                "text": entry["text"],
                "source": entry["source"],
                "category": entry["category"],
                "timestamp": entry["timestamp"],
                "verdict": entry["verdict"],
                "credibility_score": entry["credibility_score"]
            }
        ))

    operation_info = client.upsert(
        collection_name=config.COLLECTION_NAME,
        wait=True,
        points=points
    )
    print(f"Successfully indexed {len(points)} documents. Status: {operation_info.status}")

if __name__ == "__main__":
    ingest_data()