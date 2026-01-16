import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# --- UPDATED KNOWLEDGE BASE ---
KNOWLEDGE_BASE = [
    # Technical Info
    "The LiveKit Voice Agent uses a pipeline architecture with Deepgram, Groq, and Cartesia.",
    "RAG stands for Retrieval-Augmented Generation. It allows the AI to search a private dataset before answering.",
    "This agent is running on Windows using the 'uv' package manager.",
    
    # Basil Info (Since you asked!)
    "Basil is a culinary herb of the family Lamiaceae (mints).",
    "Sweet basil (Ocimum basilicum) is the most common variety used in Italian cuisine, especially pesto.",
    "Thai basil (O. basilicum var. thyrsiflora) has a distinct anise-licorice flavor and is used in Southeast Asian cooking.",
    "Holy basil (Tulsi) is widely used in India for religious and medicinal purposes.",
    "Basil grows best in warm, tropical climates and cannot survive frost.",
    "Genovese basil is the classic variety for pesto, known for its large, tender leaves.",
    "Lemon basil has a distinct citrus aroma and is often used in Indonesian cuisine."
]

class RAGEngine:
    def __init__(self):
        print("Loading embedding model...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384
        self.index = faiss.IndexFlatL2(self.dimension)
        self._build_index()

    def _build_index(self):
        embeddings = self.encoder.encode(KNOWLEDGE_BASE)
        self.index.add(np.array(embeddings))
        print(f"RAG Index built with {len(KNOWLEDGE_BASE)} documents.")

    def search(self, query: str, k: int = 3) -> str:
        print(f"--> RAG Search triggered for: '{query}'")
        vec = self.encoder.encode([query])
        distances, indices = self.index.search(np.array(vec), k)
        
        results = []
        for idx in indices[0]:
            if idx < len(KNOWLEDGE_BASE):
                results.append(KNOWLEDGE_BASE[idx])
        return "\n".join(results)