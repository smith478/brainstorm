from langchain_ollama import OllamaLLM as Ollama
from config import Config

def get_llm(config: Config):
    """Initializes and returns the LLM instance."""
    return Ollama(
        model=config.ollama_model,
        base_url=config.ollama_base_url
    )
