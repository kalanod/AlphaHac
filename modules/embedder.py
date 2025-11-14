from langchain_community.embeddings import HuggingFaceEmbeddings

# model = HuggingFaceEmbeddings(
#     model_name='ai-sage/Giga-Embeddings-instruct',
#     model_kwargs={'device': 'cuda'},
#     encode_kwargs={'normalize_embeddings': True}
# )
model = HuggingFaceEmbeddings(
    model_name='distiluse-base-multilingual-cased-v2',
    model_kwargs={'device': 'cpu', "trust_remote_code": True},
    encode_kwargs={'normalize_embeddings': True},
)


class Embedder:
    embedder = model

    def embedd_docs(self, documents):
        embeddings = self.embedder.embed_documents([i.page_content for i in documents])
        return embeddings

    def embedd_query(self, query):
        return self.embedder.execute_query(query)

    def get_embedder(self):
        return self.embedder
