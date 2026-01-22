import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer

class EvidenceRetriever:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = QdrantClient(":memory:") 
        self.collection_name = "fact_check_ledger"
        self._ensure_data_loaded()

    def _ensure_data_loaded(self):
        if not self.client.collection_exists(self.collection_name):
            print("☁️ Cloud Mode: Creating new collection...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            
            print("📂 Loading Knowledge Base...")
            try:
                with open('data/knowledge_base.json', 'r') as f:
                    data = json.load(f)
                
                documents = [d['text'] for d in data]
                embeddings = self.encoder.encode(documents)
                
                points = []
                for idx, doc in enumerate(data):
                    points.append({
                        "id": idx,
                        "vector": embeddings[idx].tolist(),
                        "payload": doc
                    })
                
                self.client.upload_points(
                    collection_name=self.collection_name,
                    points=points
                )
                print(f"✅ indexed {len(points)} facts into memory.")
            except Exception as e:
                print(f"❌ Error loading data: {e}")

    def search(self, query, k=3):
        query_vector = self.encoder.encode(query).tolist()
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=k
        )
        return hits