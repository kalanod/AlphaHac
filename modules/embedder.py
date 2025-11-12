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


    def rerank_context(self, docs):
        return docs[0][0]

    def generate_answer(self, question, context):
        return context

    def normalize_question(self, question):
        return question

    

    def new_question(self, question: str, context=""):
        question = self.normalize_question(question)
        query_embedding = self.embedd_query(question)
        results = self.search_in_chroma(query_embedding, top_k=3)
        context = context + "\n" + self.rerank_context(results)
        answer = self.generate_answer(question, context)
        return answer
