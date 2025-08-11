import os
import json
import shutil
from rest_framework.decorators import api_view
from rest_framework.response import Response
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import the RAG core files
from .config import GOOGLE_API_KEY
from .gemini_handler import GeminiEmbeddingFunction, embed_query, get_answer_from_gemini
from .chromadb_handler import ChromaDBHandler

# Global variable to hold the ChromaDB handler and documents
db_handler = None
LAST_RUN_STATE_FILE = os.path.join(os.path.dirname(__file__), '..', 'last_run_state.json')
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'chroma_db')


def initialize_rag_system():
    global db_handler
    if db_handler is None:
        print("Initializing RAG system...")

        # Check if source files have been updated
        if check_for_updates():
            print("Detected changes in data files. Re-embedding documents...")
            # Delete old database to force a rebuild
            if os.path.exists(CHROMA_DB_PATH):
                shutil.rmtree(CHROMA_DB_PATH)

        # Initialize the ChromaDB handler with a persistent directory
        db_handler = ChromaDBHandler(persist_directory=CHROMA_DB_PATH)

        # Check if the database is empty and needs to be populated
        if db_handler.collection.count() == 0:
            print("ChromaDB is empty. Loading and embedding documents...")
            directory_path = os.path.join(os.path.dirname(__file__), '..', 'data')
            documents = load_and_chunk_documents(directory_path)

            if not documents:
                print("No documents were loaded. Exiting.")
                db_handler = None  # Reset handler if no docs found
            else:
                db_handler.add_documents(documents)
                print("RAG system initialized and documents embedded.")
                save_current_state()
        else:
            print("ChromaDB already contains documents. Loading embeddings from persistent storage.")


def check_for_updates():
    """Compares current file modification times with a saved state."""
    current_state = {}
    directory_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    if not os.path.isdir(directory_path):
        return True  # Directory not found, assume it's a new run

    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        if os.path.isfile(filepath):
            current_state[filename] = os.path.getmtime(filepath)

    if not os.path.exists(LAST_RUN_STATE_FILE):
        return True  # State file doesn't exist, first run or needs update

    with open(LAST_RUN_STATE_FILE, 'r') as f:
        last_run_state = json.load(f)

    if current_state != last_run_state:
        return True  # Files have changed

    return False


def save_current_state():
    """Saves the current file modification times to a state file."""
    current_state = {}
    directory_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        if os.path.isfile(filepath):
            current_state[filename] = os.path.getmtime(filepath)

    with open(LAST_RUN_STATE_FILE, 'w') as f:
        json.dump(current_state, f)


def load_and_chunk_documents(directory_path):
    """Loads all text files from a directory and splits them into chunks."""
    all_documents = []
    if not os.path.isdir(directory_path):
        print(f"Error: Directory '{directory_path}' not found at {directory_path}.")
        return all_documents

    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        if os.path.isfile(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_text(text)
            all_documents.extend(chunks)
    return all_documents


@api_view(['POST'])
def chat_view(request):
    global db_handler
    # This block ensures the RAG system is initialized on the first request
    if db_handler is None:
        initialize_rag_system()
        if db_handler is None:
            return Response({"error": "RAG system failed to initialize."}, status=500)

    try:
        data = json.loads(request.body)
        user_question = data.get('question')

        if not user_question:
            return Response({"error": "No question provided"}, status=400)

        print(f"Received question: {user_question}")

        # Step 1: Convert question to embedding and search ChromaDB
        print("Embedding user question...")
        query_embedding = embed_query(user_question)
        print("Searching ChromaDB for relevant documents...")
        relevant_docs = db_handler.search_documents(query_embedding)

        # Step 2: Send docs + question to Gemini for an answer
        print("Generating answer with Gemini...")
        final_answer = get_answer_from_gemini(user_question, relevant_docs)
        print("Answer successfully generated.")

        return Response({"answer": final_answer})

    except Exception as e:
        print(f"An error occurred in chat_view: {e}")
        return Response({"error": str(e)}, status=500)
