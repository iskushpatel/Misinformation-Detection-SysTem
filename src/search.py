import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer


class EvidenceRetriever:
    def __init__(self):
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = QdrantClient(":memory:")
        self.collection_name = "fact_check_ledger"
        self._ensure_data_loaded()

    def _ensure_data_loaded(self):
        if self.client.collection_exists(self.collection_name):
            return

        print("☁️ Initializing in-memory knowledge base...")

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

        with open("data/knowledge_base.json", "r") as f:
            data = json.load(f)

        documents = [d["text"] for d in data]
        embeddings = self.encoder.encode(documents)

        points = [
            PointStruct(
                id=i,
                vector=embeddings[i].tolist(),
                payload=d
            )
            for i, d in enumerate(data)
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        print(f"✅ Loaded {len(points)} facts into memory")

    def search(self, user_claim, limit=5):
        query_vector = self.encoder.encode(user_claim).tolist()

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
        )

        return response.points
