import chromadb
from .gemini_handler import GeminiEmbeddingFunction

class ChromaDBHandler:
    def __init__(self, collection_name="rag_collection", persist_directory=None):
        if persist_directory:
            # Use a persistent client that saves data to disk
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            # Use an in-memory client
            self.client = chromadb.Client()

        self.embedding_function = GeminiEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

    def add_documents(self, documents):
        """Adds text documents to the ChromaDB collection."""
        ids = [f"doc_{i}" for i in range(len(documents))]
        self.collection.add(
            documents=documents,
            ids=ids
        )
        print(f"Added {len(documents)} documents to the collection.")
        # Make sure to persist changes to disk
        self.client.persist()

    def search_documents(self, query_embedding, n_results=5):
        """Searches for relevant documents using a query embedding."""
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        return results['documents'][0]
