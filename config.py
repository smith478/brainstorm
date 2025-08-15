import json
import os
from dataclasses import dataclass

@dataclass
class Config:
    """Central configuration for the system"""
    ollama_model: str = "llama3.2"  # Change to any Ollama model
    ollama_base_url: str = "http://localhost:11434"
    arxiv_max_results: int = 20
    weeks_lookback: int = 2
    db_path: str = "./brainstorm_db"
    challenges_path: str = "./challenges"
    papers_cache_path: str = "./papers_cache"
    
    def save(self, path: str = "./config.json"):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def load(cls, path: str = "./config.json"):
        if os.path.exists(path):
            with open(path, 'r') as f:
                return cls(**json.load(f))
        return cls()
