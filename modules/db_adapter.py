from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List


class DBAdapter:
    def __init__(self, embedder):
        self.persist_directory = "db"
        self.db = Chroma(
            collection_name="main",
            persist_directory=self.persist_directory,
            embedding_function=embedder,
        )

    def save_to_db(self, documents: List[Document]):
        if not documents:
            return

        batch_size = 5000
        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            self.db.add_documents(documents=batch)

    def search_in_db(self, query, top_k: int = 5):
        docs_with_scores = self.db.similarity_search_with_score(query, k=top_k)

        results = []
        for doc, score in docs_with_scores:
            doc.metadata["distance"] = score
            results.append(doc)

        return results
