from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from typing import List, Union


class DBAdapter:
    def __init__(self, persist_directory: str = "db"):
        self.persist_directory = persist_directory
        self.db = None

    def save_to_db(self, documents: List[Document], embeddings: Embeddings):
        self.db = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=self.persist_directory
        )
        self.db.persist()  # сохраняем на диск

    def search_in_db(self, query_vector: List[float], top_k: int = 5):
        results = self.db.similarity_search_by_vector(query_vector, k=top_k)

        return results
