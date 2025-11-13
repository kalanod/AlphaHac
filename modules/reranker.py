import numpy as np
from typing import List
from langchain_core.documents import Document


class Reranker:
    def __init__(self, method = "distance"):
        self.method = method
    
    def rerank_docs(self, similar_docs: List[Document], question: np.ndarray) -> List[int]:

        if len(similar_docs) == 0:
            return []
        doc_info = []
        
        for doc in similar_docs:
            doc_id = doc.metadata.get('id')
            if doc_id is None:
                continue
            distance = doc.metadata.get('distance', float('inf'))

            doc_embeddings = self._extract_doc_embeddings(doc)
            similarity = self._calculate_question_similarity(question, doc_embeddings)
            
            doc_info.append({
                'id': doc_id,
                'distance': distance,
                'similarity': similarity,
                'doc': doc
            })

        if self.method == "distance":
            ranked_docs = sorted(doc_info, key=lambda x: x['distance'])
        elif self.method == "similarity":
            ranked_docs = sorted(doc_info, key=lambda x: x['similarity'], reverse=True)
        elif self.method == "hybrid":
            ranked_docs = self._hybrid_ranking(doc_info)
        else:
            ranked_docs = sorted(doc_info, key=lambda x: x['distance'])

        top_5_ids = [doc['id'] for doc in ranked_docs[:5]]
        
        return top_5_ids
    
    def _extract_doc_embeddings(self, doc) -> np.ndarray:

        # Вариант 1: Эмбединги в page_content 
        if hasattr(doc.page_content, '__iter__') and not isinstance(doc.page_content, str):
            return np.array(doc.page_content)
        
        # Вариант 2: Эмбединги в metadata
        embeddings = doc.metadata.get('embeddings')
        if embeddings is not None:
            return np.array(embeddings)
    
    def _calculate_question_similarity(self, question: np.ndarray, doc_embeddings: np.ndarray) -> float:

        if len(doc_embeddings.shape) == 1:
            return self._cosine_similarity(question, doc_embeddings)

        max_similarity = -1
        for i in range(doc_embeddings.shape[0]):
            similarity = self._cosine_similarity(question, doc_embeddings[i])
            if similarity > max_similarity:
                max_similarity = similarity
        
        return max_similarity
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _hybrid_ranking(self, doc_info):
        distances = [doc['distance'] for doc in doc_info]
        similarities = [doc['similarity'] for doc in doc_info]
        
        min_dist, max_dist = min(distances), max(distances)
        min_sim, max_sim = min(similarities), max(similarities)

        for doc in doc_info:
            if max_dist - min_dist > 0:
                norm_distance = 1 - (doc['distance'] - min_dist) / (max_dist - min_dist)
            else:
                norm_distance = 1
            if max_sim - min_sim > 0:
                norm_similarity = (doc['similarity'] - min_sim) / (max_sim - min_sim)
            else:
                norm_similarity = 1

            doc['hybrid_score'] = 0.7 * norm_similarity + 0.3 * norm_distance
        return sorted(doc_info, key=lambda x: x['hybrid_score'], reverse=True)