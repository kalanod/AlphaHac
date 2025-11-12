from langchain_community.embeddings import HuggingFaceEmbeddings

model = HuggingFaceEmbeddings(
    model_name='ai-sage/Giga-Embeddings-instruct',
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)
class Embedder:
    embedder = model
    def embedd_docs(self, documents):
        embeddings = self.embedder.embed_documents([i.page_content for i in documents])
        return embeddings

    def embedd_query(self, query):
        return self.embedder.execute_query(query)
