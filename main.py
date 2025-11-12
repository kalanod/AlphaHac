from tqdm import tqdm

from modules.db_adapter import DBAdapter
from modules.embedder import Embedder
from modules.llm_adapter import LLMAdapter
from modules.parser import Parser
from modules.reranker import Reranker
from modules.writer import Writer

parser = Parser()
embedder = Embedder()
llm_adapter = LLMAdapter()
db_adapter = DBAdapter()
reranker = Reranker()
writer = Writer()


if __name__ == '__main__':
    documents = parser.parse_train()
    embeddings = embedder.embedd_docs(documents)
    db_adapter.save_to_db(documents, embeddings)
    questions = parser.parse_questions()
    answers = []
    for question in tqdm(questions):
        question = llm_adapter.normalize(question)
        embedded_question = embedder.embedd_query(question)
        similar_docs = db_adapter.search_in_chroma(embedded_question)
        answer = reranker.rerank_docs(similar_docs, questions)
        answers.append(answer)
    writer.write_answers(answers)

