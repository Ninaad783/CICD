import os
import time
import glob
from typing import Dict, List, Any
from google import genai
from google.genai import types
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

# Ensure GEMINI_API_KEY is configured
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("Warning: GEMINI_API_KEY environment variable is not set. Using dummy key.")
    api_key = "DUMMY_KEY"

# Initialize the new SDK client
client = genai.Client(api_key=api_key)

class GeminiEmbeddingFunction(EmbeddingFunction):
    """Custom embedding function to use Gemini embeddings inside ChromaDB."""
    def __call__(self, input: Documents) -> Embeddings:
        # Check if empty
        if not input:
            return []
        try:
            response = client.models.embed_content(
                model="gemini-embedding-2",
                contents=input
            )
            return [e.values for e in response.embeddings]
        except Exception as e:
            print(f"Error calling Gemini Embedding API: {e}")
            # Fallback to zero vectors if API fails (for testing/CI offline robustness)
            return [[0.0] * 768 for _ in input]

class RAGPipeline:
    def __init__(self, kb_dir: str = "kb", model_name: str = "gemini-3.5-flash"):
        self.kb_dir = kb_dir
        self.model_name = model_name
        self.chroma_client = chromadb.EphemeralClient()
        self.embedding_fn = GeminiEmbeddingFunction()
        self.collection = self.chroma_client.get_or_create_collection(
            name="rag_knowledge_base",
            embedding_function=self.embedding_fn
        )
        
    def load_and_index_knowledge_base(self):
        """Reads all txt files from the kb directory and adds them to ChromaDB."""
        if not os.path.exists(self.kb_dir):
            os.makedirs(self.kb_dir)
            # Create a default sample file if directory is empty
            with open(os.path.join(self.kb_dir, "sample.txt"), "w", encoding="utf-8") as f:
                f.write("Antigravity is a coding assistant built by the Google DeepMind team. "
                        "It is designed to help software engineers with complex codebase modifications, "
                        "bug fixes, and architectural planning using agentic frameworks.")
        
        txt_files = glob.glob(os.path.join(self.kb_dir, "*.txt"))
        documents = []
        metadatas = []
        ids = []
        
        for idx, file_path in enumerate(txt_files):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        # For simple demo, we treat each file as one chunk.
                        # In production, we would chunk files.
                        documents.append(content)
                        metadatas.append({"source": os.path.basename(file_path)})
                        ids.append(f"doc_{idx}")
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                
        if documents:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Successfully indexed {len(documents)} documents in ChromaDB.")
        else:
            print("No documents found in knowledge base directory to index.")

    def query(self, user_query: str, n_results: int = 2) -> Dict[str, Any]:
        """Queries ChromaDB and generates an answer using Gemini."""
        start_time = time.time()
        
        # 1. Retrieve relevant chunks
        try:
            results = self.collection.query(
                query_texts=[user_query],
                n_results=n_results
            )
            retrieved_contexts = results['documents'][0] if results['documents'] else []
            sources = [meta.get('source', 'unknown') for meta in results['metadatas'][0]] if results['metadatas'] else []
        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
            retrieved_contexts = []
            sources = []

        context_str = "\n---\n".join(retrieved_contexts)
        
        # 2. Formulate Prompt
        prompt = f"""You are an assistant answering questions based ONLY on the provided context.
If the context does not contain the answer, say "I do not have enough information to answer this."

Context:
{context_str}

Question:
{user_query}

Answer:"""

        # 3. Generate Answer using Gemini
        answer = ""
        try:
            if os.environ.get("MOCK_EVAL") == "true":
                answer = f"Mock answer for: {user_query} (using context: {sources})."
            else:
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0  # low temperature for factual answers
                    )
                )
                answer = response.text.strip()
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print("  -> Quota exceeded. Falling back to mock generation.")
                answer = f"Mock answer for: {user_query} (Quota exceeded fallback)."
            else:
                answer = f"Error generating answer: {e}"
            
        latency = time.time() - start_time
        
        return {
            "query": user_query,
            "answer": answer,
            "contexts": retrieved_contexts,
            "sources": sources,
            "latency": latency
        }

if __name__ == "__main__":
    # Test execution
    pipeline = RAGPipeline()
    pipeline.load_and_index_knowledge_base()
    res = pipeline.query("What is Antigravity?")
    print("\n--- Test RAG Output ---")
    print(f"Query: {res['query']}")
    print(f"Answer: {res['answer']}")
    print(f"Latency: {res['latency']:.2f}s")
