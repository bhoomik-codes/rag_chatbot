import google.generativeai as genai
from .config import GOOGLE_API_KEY

genai.configure(api_key=GOOGLE_API_KEY)


# Custom embedding function for ChromaDB
class GeminiEmbeddingFunction:

    def name(self):
        """Returns the name of the embedding function."""
        return "gemini-embedding-function"

    def __call__(self, input):
        return [self._embed_text(text) for text in input]

    def _embed_text(self, text):
        return genai.embed_content(model="models/embedding-001",
                                   content=text,
                                   task_type="retrieval_document")['embedding']


def embed_query(query):
    """Generates an embedding for a user query."""
    return genai.embed_content(model="models/embedding-001",
                               content=query,
                               task_type='retrieval_query')['embedding']


def get_answer_from_gemini(question, context_docs):
    """Sends question + context to Gemini to get an answer."""
    # Using the default model to ensure compatibility with your environment
    model = genai.GenerativeModel('gemini-pro')

    # Construct the prompt
    prompt = f"""
    You are a helpful assistant. Use the following documents to answer the user's question.
    If the answer is not in the documents, say "I couldn't find the answer in the provided documents."

    Documents:
    {chr(10).join(context_docs)}

    Question: {question}

    Answer:
    """

    response = model.generate_content(prompt)
    return response.text
