import openai
import time
from .base import LLM
import asyncio

class OpenAiLlm(LLM):
    def __init__(self, api_key: str, prompt: str, model:str="gpt-3.5-turbo-1106", tools=None):
        super().__init__(api_key, prompt, model)
        if tools is None:
            tools = [{"type": "retrieval"}]
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.assistant = self.client.beta.assistants.create(
            instructions=self.instruction,
            model=self.model,
            tools=tools,
        )
        
    async def create_thread(self):
        return self.client.beta.threads.create()

    async def create_message(self, content: str, thread: object):
        self.message = await self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=content,
        )
    
    async def get_response(self, content: str, thread: object=None):
        thread = await self.create_thread()
        self.message = await self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=content,
        )
        runner = await self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=self.assistant.id
        )
        await self.client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=runner.id
        )
        start = time.perf_counter()
        while True:
            messages = await self.client.beta.threads.messages.list(
                thread_id=thread.id
            )
            current = time.perf_counter()
            if len(messages.data) > 1:
                print(f"get response from openai finishes in {current - start}")
                break
            if current - start > 10:
                print("get response time out")
                break
        message = await self.client.beta.threads.messages.list(thread_id=thread.id).data[-1]
        print(message.content[0].text.value)
        return message.content[0].text.value