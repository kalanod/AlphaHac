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
db_adapter = DBAdapter(embedder.get_embedder())
reranker = Reranker()
writer = Writer()


if __name__ == '__main__':
    documents = parser.parse_train()
    db_adapter.save_to_db(documents)
    questions = parser.parse_questions()
    answers = []
    for question in tqdm(questions):
        question = llm_adapter.normalize(question)
        similar_docs = db_adapter.search_in_db(question)
        answer = reranker.rerank_docs(similar_docs, questions)
        answers.append(answer)
    writer.write_answers(answers)

