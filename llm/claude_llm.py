import openai
import time
from .base import LLM
import asyncio
import aiohttp
API_URL = "https://api.anthropic.com/v1/complete"
import anthropic

class ClaudeLLM(LLM):
    def __init__(self, api_key: str, prompt: str, model:str="gpt-3.5-turbo-1106", tools=None):
        super().__init__(api_key, prompt, model)
        if tools is None:
            tools = [{"type": "retrieval"}]
        self.api_key = api_key
    
    async def get_response(self, content: str, thread: object=None):
        client = anthropic.Client(self.api_key)

        conversation_history = '\n'.join(self.conversation_history)
        prompt = f"{conversation_history}\nHuman: {prompt}\nAssistant: "

        response = await client.acompletion(
            prompt=prompt,
            stop_sequences=["\n\nHuman:"],
            max_tokens_to_sample=150,
            model="claude-v1",
            temperature=0.7,
        )
        
        return response.completion.strip()

