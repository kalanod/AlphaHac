from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from core.RagContext import RagContext

MAIN_EMBEDDER = HuggingFaceEmbeddings(
    model_name='ai-sage/Giga-Embeddings-instruct',
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

if __name__ == '__main__':
    context = RagContext("main", MAIN_EMBEDDER)
    context.add_file(Path("./res/diapi.pdf"))
    context.new_question("сколько будут платить?")