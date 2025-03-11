import json

from pymongo import MongoClient
import requests
import time
import re
import tiktoken
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
    LengthBasedExampleSelector
)
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def convert_to_2d_array(input_string):
    chunks = re.split(r'<START>', input_string)
    result = []

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        lines = re.findall(r'{{user}}:(.*?)(?={{char}}:|$)|{{char}}:(.*?)(?={{user}}:|$)', chunk, re.DOTALL)
        lines = [line.strip() for user_line, char_line in lines for line in (user_line, char_line) if line.strip()]

        if lines:
            result.append(lines)

    return result


class SillyTavern:
    def __init__(self, url='http://localhost:6999'):
        self.url = url

    def create_character(self, form_data):
        url = self.url + "/api/characters/create"
        print(url)
        response = requests.post(url, data=form_data)
        if response.status_code != 200:
            print(response)
            return None
        return response.text[:-4]

    def get_character_info(self, assistant_id):
        url = self.url + "/api/characters/get"
        response = requests.post(url, json={"avatar_url": f"{assistant_id}.png"})
        if response.status_code != 200:
            return None
        return response.json()

    def edit_character_info(self, name, data):
        pass

    def save_thread(self, assistant_id, user_id, thread_id, ch_name, thread_info):
        url = self.url + "/api/chats/save"
        file_name = f"{assistant_id}_{user_id}_{thread_id}"
        payload = {
            "avatar_url": f"{assistant_id}.png",
            "ch_name": ch_name,
            "chat": thread_info,
            "file_name": file_name
        }
        response = requests.post(url, json=payload)
        if response.status_code == 200 and response.json().get("result") == "ok":
            return file_name
        return None

    def upload_thread(self, file_name, user_id, assistant_id):
        pass

    def create_thread(self, assistant_id, user_id, user_name):
        character_info = self.get_character_info(assistant_id)
        first_msg = character_info['data']['first_mes']
        name = character_info['name']
        current_time_ms = int(time.time() * 1000)
        # save character data
        # find username from user_id
        file_name = f"{assistant_id}_{user_id}_{current_time_ms}"
        url = self.url + "/api/chats/save"
        payload = {
            "avatar_url": f"{assistant_id}.png",
            "ch_name": name,
            "chat": [
                {
                    "character_name": name,
                    "chat_metadata": {},
                    "create_date": current_time_ms,
                    "user_name": user_name,
                },
                {
                    "extra": {},
                    "is_system": False,
                    "is_user": False,
                    "mes": first_msg,
                    "name": name,
                    "send_date": current_time_ms,
                }
            ],
            "file_name": file_name
        }
        response = requests.post(url, json=payload)
        if response.status_code == 200 and response.json().get("result") == "ok":
            return file_name
        return None

    def messages_parser(self, thread_id, user_id, assistant_id, user_name, query, maximum_token=24000):
        total_tokens = 0

        def num_tokens_from_string(string: str, encoding_name: str = 'cl100k_base') -> int:
            """Returns the number of tokens in a text string."""
            encoding = tiktoken.get_encoding(encoding_name)
            num_tokens = len(encoding.encode(string))
            return num_tokens

        messages = []
        character_info = self.get_character_info(assistant_id)
        if character_info is None:
            return None
        if character_info.get('system_prompt', '') != '':
            messages.append({'role': 'system', 'content': character_info.get('system_prompt', '')})
            total_tokens += num_tokens_from_string(character_info.get('system_prompt', ''), 'cl100k_base')
        else:
            messages.append({'role': 'system', 'content': system_prompt})
            total_tokens += num_tokens_from_string(system_prompt, 'cl100k_base')
        world = character_info.get('extensions', {}).get('world', '')
        if world != '':
            content = self.retrieve_world_setting(query.get('text'), world)
            messages.append({'role': 'system', 'content': content})
            total_tokens += num_tokens_from_string(content, 'cl100k_base')
        messages.append({'role': 'system', 'content': character_info.get('description', '')})
        total_tokens += num_tokens_from_string(character_info.get('description', ''), 'cl100k_base')
        if character_info.get('personality', '') != '':
            messages.append(
                {'role': 'system', 'content': f"[Arya2's personality: {character_info.get('personality', '')}]"})
            total_tokens += num_tokens_from_string(character_info.get('personality', ''), 'cl100k_base')

        scenario = character_info.get('scenario', '')
        if scenario != '':
            messages.append({'role': 'system', 'content': f"[Circumstances and context of the dialogue: {scenario}]"})
            total_tokens += num_tokens_from_string(scenario, 'cl100k_base')
        messages.append({'role': 'system', 'content': nsfw_prompt})
        total_tokens += num_tokens_from_string(nsfw_prompt, 'cl100k_base')
        mes_example = character_info.get('mes_example', '')
        if mes_example != '':
            mes_example = convert_to_2d_array(mes_example)
            for example in mes_example:
                messages.append({'role': 'system', 'content': "[Example Chat]"})
                total_tokens += num_tokens_from_string("[Example Chat]", 'cl100k_base')
                for i, mes in enumerate(example):
                    if i % 2 == 0:
                        messages.append({'role': 'system', 'content': mes, 'name': "example_user"})
                    else:
                        messages.append({'role': 'system', 'content': mes, 'name': "example_assistant"})
                    total_tokens += num_tokens_from_string(mes, 'cl100k_base')
        messages.append({'role': 'system', 'content': "[Start a new Chat]"})
        thread = self.get_thread_info(assistant_id, thread_id, user_id)
        if thread is not None:
            def extract_name_and_mes(data):
                return {
                    "name": data["name"],
                    "mes": data["mes"]
                }

            extracted_data = [extract_name_and_mes(item) for item in thread[1:]]

            example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
                # The list of examples available to select from.
                extracted_data,
                # The embedding class used to produce embeddings which are used to measure semantic similarity.
                OpenAIEmbeddings(api_key=''),
                # The VectorStore class that is used to store the embeddings and do a similarity search over.
                FAISS,
                # The number of examples to produce.
                k=9999,
            )
            thread_messages = example_selector.select_examples({'name': user_name, 'mes': query.get('text', '')})
            remove_list = []
            for mes in thread_messages:
                total_tokens += num_tokens_from_string(mes.get('mes', ''), 'cl100k_base')
                if total_tokens > maximum_token:
                    remove_list.append(mes['mes'])

            for mes in thread[1:]:
                if mes['mes'] in remove_list:
                    continue
                if mes.get('is_user', False):
                    messages.append({'role': 'user', 'content': mes.get('mes', '')})
                elif mes.get('is_system', False):
                    messages.append({'role': 'system', 'content': mes.get('mes', '')})
                else:
                    messages.append({'role': 'assistant', 'content': mes.get('mes', '')})
        else:
            messages.append({'role': 'assistant', 'content': character_info.get('first_mes', '')})
        messages.append({'role': 'user', 'content': query.get('text', '')})
        messages.append({'role': 'system', 'content': jailbreak_prompt})
        for mes in messages:
            mes['content'] = mes['content'].replace("{{user}}", user_name).replace("{{char}}", character_info['name'])
        return messages

    def chat_completion(self, thread_id, user_id, assistant_id, query, user_name, generation_params):
        messages = self.messages_parser(thread_id, user_id, assistant_id, user_name, query)
        url = self.url + "/api/backends/chat-completions/generate"
        payload = {
            "messages": messages,
            **generation_params
        }
        payload = json.dumps(payload)
        headers = {
            "Content-Type": "application/json",
        }
        response = requests.post(url, data=payload.encode('utf-8'), stream=True, headers=headers)
        for line in response.iter_lines():
            if line:
                try:
                    data_dict = json.loads(line.decode('utf-8')[6:])
                    yield data_dict['choices'][0]['delta']['content']
                except:
                    pass

    def create_world(self, user_id, assistant_id):
        pass

    def get_world_info(self, name):
        url = self.url + "/api/worldinfo/get"
        response = requests.post(url, json={"name": name})
        if response.status_code != 200:
            return None
        return response.json()

    def retrieve_world_setting(self, string, name):
        content = ""
        world_info = self.get_world_info(name)['entries']
        max_idx = len(world_info)
        i = 0
        while i < max_idx:
            entry = world_info.get(str(i), '')
            if entry == '':
                i += 1
                continue
            flag = False
            for keyword in entry['key']:
                if keyword.lower() in string.lower() or keyword.lower() in content.lower():
                    content += entry['content']
                    content += '\n'
                    del world_info[str(i)]
                    flag = True
                    break
            if flag:
                i = 0
            else:
                i += 1
        return f"[Details of the fictional world the RP is set in:{content}]\n"

    def edit_world_info(self, name, data):
        pass

    def get_thread_info(self, assistant_id, thread_id, user_id):
        filename = f"{assistant_id}_{user_id}_{thread_id}"
        character_info = self.get_character_info(assistant_id)
        name = character_info['name']
        avatar_url = f"{assistant_id}.png"
        url = self.url + "/api/chats/get"
        response = requests.post(url, json={"avatar_url": avatar_url, "ch_name": name, "file_name": filename})
        if response.status_code != 200:
            return None
        return response.json()

    def edit_thread_info(self, assistant_id, thread_id, user_id):
        pass
