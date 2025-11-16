import numpy as np
from typing import List, Optional
from langchain_core.documents import Document


class Reranker:
    def __init__(self, method="distance"):
        self.method = method

    def rerank_docs(self, similar_docs: List[Document]) -> List[int]:
        if len(similar_docs) == 0:
            return []
        doc_distances = []
        
        for doc in similar_docs:
            doc_id = (
                doc.metadata.get('web_id')
                or doc.metadata.get('id')
                or doc.metadata.get('source')
            )
            if doc_id is None:
                continue
                
            distance = doc.metadata.get('distance', float('inf'))
            doc_distances.append((doc_id, distance))
        
        if not doc_distances:
            return []
        ranked_docs = sorted(doc_distances, key=lambda x: x[1])
        top_5_ids = [doc_id for doc_id, distance in ranked_docs[:5]]
        
        return top_5_ids
