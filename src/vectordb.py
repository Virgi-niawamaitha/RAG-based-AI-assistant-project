# Standard library
import os
from typing import List, Dict, Any

# Third-party
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """

    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace model name for embeddings
        """
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Always resolve path relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        chroma_path = os.path.join(project_root, "chroma_db")
        self.client = chromadb.PersistentClient(path=chroma_path)

        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

        print(f"Vector database initialized with collection: {self.collection_name}")

    def chunk_text(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
        """
        Split text into chunks using LangChain's RecursiveCharacterTextSplitter.
        
        Args:
            text: Input text to chunk
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        
        Returns:
            List of text chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_text(text)

    def add_documents(self, documents: List[Dict], verbose: bool = False) -> None:
        """
        Add documents to the vector database.
        
        Args:
            documents: List of dicts with 'content' and 'metadata' keys
            verbose: If True, print detailed progress information
        """
        print(f"Processing {len(documents)} documents...")
        
        for doc_idx, doc in enumerate(documents):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})

            chunks = self.chunk_text(content)
            
            if verbose:
                print(f"  Doc {doc_idx}: {len(chunks)} chunks")
            
            if len(chunks) == 0:
                print(f"WARNING: No chunks generated for document {doc_idx}")
                continue

            chunk_ids = [f"doc_{doc_idx}_chunk_{i}" for i in range(len(chunks))]
            embeddings = self.embedding_model.encode(chunks).tolist()

            self.collection.add(
                documents=chunks,
                metadatas=[metadata] * len(chunks),
                ids=chunk_ids,
                embeddings=embeddings
            )
            
            if verbose:
                print(f"  Added {len(chunks)} chunks to database")

        print("Documents added to vector database")

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database.

        Args:
            query: User's search query
            n_results: Number of most similar chunks to return

        Returns:
            Dictionary with keys: 'documents', 'metadatas', 'distances', 'ids'
        """
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        return {
            "documents": results.get("documents", [[]])[0],
            "metadatas": results.get("metadatas", [[]])[0],
            "distances": results.get("distances", [[]])[0],
            "ids": results.get("ids", [[]])[0],
        }