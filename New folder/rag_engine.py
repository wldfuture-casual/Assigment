import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer


class RAGEngine:
    """Retrieval Augmented Generation engine using embeddings"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize with embedding model"""
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.chunks = []
        self.embeddings = None
        print("✓ RAG engine ready")
    
    def chunk_text(self, text: str, min_chunk_size: int = 50) -> List[str]:
        """
        Split text into semantic chunks
        
        Args:
            text: Input text to chunk
            min_chunk_size: Minimum characters per chunk
            
        Returns:
            List of text chunks
        """
        # Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Further split long paragraphs by sentences
        chunks = []
        for para in paragraphs:
            if len(para) < min_chunk_size:
                # Combine small paragraphs
                if chunks and len(chunks[-1]) < 200:
                    chunks[-1] += ' ' + para
                else:
                    chunks.append(para)
            elif len(para) > 500:
                # Split long paragraphs by sentences
                sentences = [s.strip() + '.' for s in para.split('.') if s.strip()]
                current_chunk = ''
                for sent in sentences:
                    if len(current_chunk) + len(sent) < 300:
                        current_chunk += ' ' + sent
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sent
                if current_chunk:
                    chunks.append(current_chunk.strip())
            else:
                chunks.append(para)
        
        # Filter out very small chunks
        chunks = [c for c in chunks if len(c) >= min_chunk_size]
        
        return chunks
    
    def build_index(self, chunks: List[str]) -> None:
        """
        Build embedding index from chunks
        
        Args:
            chunks: List of text chunks to embed
        """
        self.chunks = chunks
        if not chunks:
            self.embeddings = np.array([])
            return
        
        print(f"Embedding {len(chunks)} chunks...")
        self.embeddings = self.model.encode(
            chunks,
            convert_to_tensor=False,
            show_progress_bar=False
        )
        print(f"✓ Built index with {len(chunks)} chunks")
    
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """
        Retrieve top-k most relevant chunks
        
        Args:
            query: Search query
            k: Number of chunks to retrieve
            
        Returns:
            List of most relevant chunks
        """
        if not self.chunks or self.embeddings is None or len(self.embeddings) == 0:
            return []
        
        # Encode query
        query_embedding = self.model.encode(query, convert_to_tensor=False)
        
        # Calculate cosine similarities
        similarities = self._cosine_similarity(query_embedding, self.embeddings)
        
        # Get top-k indices
        k = min(k, len(self.chunks))
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        # Return chunks in relevance order
        return [self.chunks[i] for i in top_indices]
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between vectors"""
        if vec2.ndim == 1:
            vec2 = vec2.reshape(1, -1)
        
        dot_product = np.dot(vec2, vec1)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2, axis=1)
        
        return dot_product / (norm1 * norm2 + 1e-10)
    
    def get_stats(self):
        """Return indexing statistics"""
        return {
            'num_chunks': len(self.chunks),
            'embedding_dim': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'model': self.model.get_sentence_embedding_dimension()
        }


if __name__ == '__main__':
    # Test RAG engine
    rag = RAGEngine()
    
    sample_text = """
    Photosynthesis is the process by which plants convert light energy into chemical energy.
    It occurs in chloroplasts and requires sunlight, water, and carbon dioxide.
    
    The process has two main stages: light-dependent reactions and light-independent reactions.
    Light-dependent reactions occur in the thylakoid membranes and produce ATP and NADPH.
    
    Light-independent reactions, also known as the Calvin cycle, occur in the stroma.
    These reactions use ATP and NADPH to convert carbon dioxide into glucose.
    
    Photosynthesis is crucial for life on Earth as it produces oxygen and organic compounds.
    Without it, most life forms would not be able to survive.
    """
    
    chunks = rag.chunk_text(sample_text)
    print(f"\nChunked into {len(chunks)} pieces:")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk[:80]}...")
    
    rag.build_index(chunks)
    
    query = "What happens in the Calvin cycle?"
    results = rag.retrieve(query, k=2)
    
    print(f"\nQuery: {query}")
    print("Top results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result}")