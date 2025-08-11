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
        # Removed the self.client.persist() call as it's deprecated
        # and not needed with a PersistentClient in the latest versions.

    def search_documents(self, query_embedding, n_results=5):
        """Searches for relevant documents using a query embedding."""
        try:
            print(f"Searching ChromaDB with n_results={n_results}...")
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results
            )
            print(f"Found {len(results.get('documents', [[]])[0])} relevant documents.")
            return results['documents'][0]
        except Exception as e:
            print(f"Error during ChromaDB search: {e}")
            raise e
