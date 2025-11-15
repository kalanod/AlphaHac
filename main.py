from tqdm import tqdm

from modules.db_adapter import DBAdapter
from modules.embedder import Embedder
from modules.llm_adapter import LLMAdapter
from modules.parser import Parser
from modules.reranker import Reranker
from modules.writer import Writer

parser = Parser(path='res')
embedder = Embedder()
llm_adapter = LLMAdapter()
db_adapter = DBAdapter(embedder.get_embedder())
reranker = Reranker()
writer = Writer()


if __name__ == '__main__':
    documents = parser.parse_train()
    db_adapter.save_to_db(documents)
    questions = parser.parse_questions()
    answers = []
    for q_id, raw_query in tqdm(questions):
        normalized_query = llm_adapter.normalize(raw_query)
        search_query = normalized_query or raw_query
        similar_docs = db_adapter.search_in_db(search_query)
        answer = reranker.rerank_docs(similar_docs)
        answers.append((q_id, answer))
    writer.write_answers(answers)

