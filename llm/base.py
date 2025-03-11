from abc import ABC, abstractmethod
class LLM(ABC):
    def __init__(self, api_key: str, prompt: str, model: str=None):
        self.conversation_history = []
        self.api_key = api_key
        self.prompt = prompt
        self.model= model