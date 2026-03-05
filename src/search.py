"""
Retrieval Engine for FactChk RAG Application.

This module handles loading the LIAR dataset, vectorizing statements,
and storing them in a persistent Qdrant vector database for similarity search.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """Configuration for the retrieval engine."""
    
    qdrant_path: str = "qdrant_db"
    collection_name: str = "liar_statements"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_size: int = 384  # Output dimension of all-MiniLM-L6-v2
    top_k: int = 5
    similarity_threshold: float = 0.5


class RetrieverEngine:
    """
    Retrieves similar historical claims from the LIAR dataset using vector similarity.
    
    This engine loads the LIAR dataset, vectorizes statements using Sentence Transformers,
    and stores them in a persistent Qdrant vector database. It provides methods to
    search for similar claims and retrieve their metadata.
    
    Attributes:
        config: Configuration parameters for retrieval
        client: Qdrant client instance
        model: Sentence transformer model for embedding generation
        _collection_exists: Internal flag to track collection existence
    """
    
    def __init__(self, config: Optional[RetrievalConfig] = None) -> None:
        """
        Initialize the Retrieval Engine.
        
        Args:
            config: RetrievalConfig instance. If None, uses default config.
            
        Raises:
            ValueError: If the embedding model cannot be loaded.
            RuntimeError: If Qdrant connection fails.
        """
        self.config = config or RetrievalConfig()
        
        logger.info(f"Initializing RetrieverEngine with config: {self.config}")
        
        # Initialize Qdrant client with persistent storage
        try:
            self.client = QdrantClient(path=self.config.qdrant_path)
            logger.info(f"Qdrant client initialized at path: {self.config.qdrant_path}")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise RuntimeError(f"Qdrant initialization failed: {e}") from e
        
        # Load sentence transformer model
        try:
            self.model = SentenceTransformer(self.config.embedding_model)
            logger.info(f"Loaded embedding model: {self.config.embedding_model}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise ValueError(f"Model loading failed: {e}") from e
        
        self._collection_exists = self._check_collection_exists()
    
    def _check_collection_exists(self) -> bool:
        """
        Check if the collection already exists in Qdrant.
        
        Returns:
            True if collection exists, False otherwise.
        """
        try:
            collections = self.client.get_collections()
            exists = any(
                collection.name == self.config.collection_name
                for collection in collections.collections
            )
            logger.info(f"Collection '{self.config.collection_name}' exists: {exists}")
            return exists
        except Exception as e:
            logger.warning(f"Error checking collection existence: {e}")
            return False
    
    def build_database(self, dataset_path: str = "data/liar_train.tsv") -> None:
        """
        Load LIAR dataset, vectorize statements, and store in Qdrant.
        
        This method:
        1. Loads the TSV file with proper column assignment
        2. Limits to first 1000 rows for development speed
        3. Creates Qdrant collection if it doesn't exist
        4. Vectorizes statements using Sentence Transformers
        5. Stores vectors with rich metadata
        
        Args:
            dataset_path: Path to the LIAR TSV file.
            
        Raises:
            FileNotFoundError: If dataset file doesn't exist.
            ValueError: If dataset is empty or invalid.
            RuntimeError: If database building fails.
        """
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            logger.error(f"Dataset file not found: {dataset_path}")
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
        logger.info(f"Loading dataset from {dataset_path}")
        
        # Define column names for LIAR dataset
        column_names = [
            'id', 'label', 'statement', 'subjects', 'speaker',
            'job', 'state', 'party', 'bt', 'fc', 'ht', 'mt', 'pof', 'context'
        ]
        
        try:
            # Load TSV file without headers
            df = pd.read_csv(
                dataset_path,
                sep='\t',
                header=None,
                names=column_names,
                dtype={'statement': str, 'label': str, 'speaker': str, 'context': str}
            )
            
            # Limit to first 1000 rows for development speed
            df = df.head(1000)
            logger.info(f"Loaded {len(df)} statements from dataset")
            
            if df.empty:
                raise ValueError("Dataset is empty")
            
            # Create Qdrant collection
            self._create_collection()
            
            # Vectorize and store statements
            self._vectorize_and_store(df)
            
            logger.info("Database build completed successfully")
            self._collection_exists = True
            
        except Exception as e:
            logger.error(f"Failed to build database: {e}")
            raise RuntimeError(f"Database building failed: {e}") from e
    
    def _create_collection(self) -> None:
        """
        Create a new collection in Qdrant if it doesn't exist.
        
        Raises:
            RuntimeError: If collection creation fails.
        """
        try:
            # Check if collection already exists
            collections = self.client.get_collections()
            collection_exists = any(
                collection.name == self.config.collection_name
                for collection in collections.collections
            )
            
            if collection_exists:
                logger.info(f"Collection '{self.config.collection_name}' already exists, skipping creation")
                return
            
            # Create if doesn't exist
            self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=self.config.vector_size,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection: {self.config.collection_name}")
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise RuntimeError(f"Collection creation failed: {e}") from e
    
    def _vectorize_and_store(self, df: pd.DataFrame) -> None:
        """
        Vectorize statements and store them in Qdrant with metadata.
        
        Args:
            df: DataFrame containing the LIAR dataset.
            
        Raises:
            RuntimeError: If vectorization or storage fails.
        """
        logger.info("Vectorizing statements...")
        
        try:
            # Vectorize all statements
            statements = df['statement'].tolist()
            embeddings = self.model.encode(statements, show_progress_bar=True)
            logger.info(f"Generated {len(embeddings)} embeddings")
            
            # Prepare points for Qdrant
            points = []
            for idx, (_, row) in enumerate(df.iterrows()):
                point = PointStruct(
                    id=idx,
                    vector=embeddings[idx].tolist(),
                    payload={
                        'text': str(row['statement']),
                        'label': str(row['label']),
                        'speaker': str(row['speaker']),
                        'context': str(row['context']) if pd.notna(row['context']) else "",
                        'subjects': str(row['subjects']) if pd.notna(row['subjects']) else "",
                        'party': str(row['party']) if pd.notna(row['party']) else "",
                        'state': str(row['state']) if pd.notna(row['state']) else "",
                        'job': str(row['job']) if pd.notna(row['job']) else "",
                    }
                )
                points.append(point)
            
            # Upload points to Qdrant
            self.client.upsert(
                collection_name=self.config.collection_name,
                points=points
            )
            
            logger.info(f"Stored {len(points)} points in Qdrant")
            
        except Exception as e:
            logger.error(f"Failed to vectorize and store: {e}")
            raise RuntimeError(f"Vectorization failed: {e}") from e
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most similar statements from the database.
        
        Args:
            query: The user's claim to find similar statements for.
            top_k: Number of results to return. If None, uses config default.
            
        Returns:
            List of dictionaries containing retrieved statements and metadata.
            Each dictionary has keys: 'text', 'label', 'speaker', 'context', 'score'
            
        Raises:
            ValueError: If query is empty.
            RuntimeError: If retrieval fails.
            
        Example:
            >>> results = retriever.retrieve("The earth is flat", top_k=3)
            >>> for result in results:
            ...     print(f"Score: {result['score']:.2f}")
            ...     print(f"Statement: {result['text']}")
            ...     print(f"Rating: {result['label']}")
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        top_k = top_k or self.config.top_k
        
        logger.info(f"Retrieving top-{top_k} similar statements for: {query[:100]}...")
        
        try:
            # Check if collection exists
            if not self._collection_exists:
                logger.error(f"Collection '{self.config.collection_name}' does not exist!")
                logger.error("Database might not have been built. Trying to build now...")
                self.build_database()
            
            # Generate embedding for the query
            logger.debug(f"Vectorizing query...")
            query_embedding = self.model.encode(query)
            logger.debug(f"Query embedding shape: {query_embedding.shape}")
            
            # Search in Qdrant
            logger.debug(f"Searching Qdrant with similarity_threshold={self.config.similarity_threshold}")
            search_results = self.client.search(
                collection_name=self.config.collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k,
                score_threshold=self.config.similarity_threshold
            )
            
            logger.info(f"Search returned {len(search_results)} results")
            
            # Format results
            retrieved = []
            for point in search_results:
                result = {
                    'text': point.payload.get('text', ''),
                    'label': point.payload.get('label', 'unknown'),
                    'speaker': point.payload.get('speaker', 'unknown'),
                    'context': point.payload.get('context', ''),
                    'subjects': point.payload.get('subjects', ''),
                    'party': point.payload.get('party', ''),
                    'state': point.payload.get('state', ''),
                    'job': point.payload.get('job', ''),
                    'score': point.score
                }
                retrieved.append(result)
            
            logger.info(f"Retrieved {len(retrieved)} similar statements")
            return retrieved
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise RuntimeError(f"Search failed: {e}") from e
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the current collection.
        
        Returns:
            Dictionary containing collection statistics.
            
        Raises:
            RuntimeError: If collection doesn't exist or query fails.
        """
        if not self._collection_exists:
            raise RuntimeError(f"Collection '{self.config.collection_name}' does not exist")
        
        try:
            collection_info = self.client.get_collection(
                collection_name=self.config.collection_name
            )
            return {
                'collection_name': self.config.collection_name,
                'points_count': collection_info.points_count,
                'vector_size': self.config.vector_size,
                'distance_metric': 'cosine',
                'model': self.config.embedding_model
            }
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            raise RuntimeError(f"Statistics retrieval failed: {e}") from e


def get_retriever(config: Optional[RetrievalConfig] = None) -> RetrieverEngine:
    """
    Factory function to get or create a RetrieverEngine instance.
    
    This is useful for Streamlit app caching with @st.cache_resource.
    
    Args:
        config: RetrievalConfig instance. If None, uses default config.
        
    Returns:
        Initialized RetrieverEngine instance.
    """
    return RetrieverEngine(config)