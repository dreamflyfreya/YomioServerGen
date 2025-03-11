import openai
from .prompts import *

git client = openai.OpenAI(api_key="")


class RaidenShogun:
    def __init__(self):
        self.nsfw_system_prompt = nsfw_Raiden_Shogun_prompt_text
        self.system_prompt = Raiden_Shogun_prompt_text
        self.first_response = Raiden_Shogun_first_response
        self.nsfw_first_response = nsfw_Raiden_Shogun_first_response
        self.name = "Raiden Shogun"
        pass

    def respond(self, history, nsfw, user):
        if nsfw:
            messages = [{'role': 'system', 'content': self.nsfw_system_prompt},
                        {'role': 'user', 'content': 'Hello! Raiden Shogun.'},
                        {'role': 'assistant', 'content': self.nsfw_first_response}]
        else:
            messages = [{'role': 'system', 'content': self.system_prompt},
                        {'role': 'user', 'content': 'Hello! Raiden Shogun.'},
                        {'role': 'assistant', 'content': self.first_response}]

        for message in history:
            if message['name'] == self.name:
                messages.append({'role': 'assistant', 'content': message['message']})
            else:
                messages.append({'role': 'user', 'content': message['message']})

        response = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=messages,
            stream=True  # this time, we set stream=True
        )
        for chunk in response:
            yield f"data: {chunk.choices[0].delta.content}\n\n"