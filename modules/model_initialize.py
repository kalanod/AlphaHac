from llama_cpp import Llama

class ModelInitializer:
    def __init__(self, model_path: str, n_gpu_layers: int = 20, n_ctx: int = 4096):
        """
        Args:
            model_path: путь к GGUF модели
            n_gpu_layers: количество слоев для загрузки на GPU
            n_ctx: размер контекста
        """
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False
        )
        print(f"Модель {model_path} успешно загружена")
    
    def get_model(self):
        return self.llm