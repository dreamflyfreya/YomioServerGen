import base64
import json
import os
import time

import pydantic
import tiktoken
import httpx
import asyncio

from langchain import hub
from langchain_community.chat_models import BedrockChat
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import BaseMessage
from langchain_community.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain.chains import create_extraction_chain_pydantic
from langchain_core.pydantic_v1 import BaseModel
from langchain import hub
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from typing import List
import httpx_cache

from .prompts import *
from config import *
from .character_config import *
from .create_model import create_model
from .get_knowledge_retrieval import *
memory_lock = asyncio.Lock()
client = MongoClient(mongo_uri, int(mongo_port))
uri = f"mongodb://{mongo_uri}:{mongo_port}"
db = client.solaris

persona_knowledge_retriever, world_info_retriever = initialize_knowledge_retriever(character_config_list)

memory = MongoDBChatMessageHistory(
    session_id=0,
    connection_string=uri,
    database_name="solaris",
    collection_name="chat_histories",
)

testing_time = 0

chromadb_persist_directory = os.path.expanduser('~/chromadb')
embeddings = OpenAIEmbeddings(model='text-embedding-3-large', openai_api_key=openai_key)

persona_collection = db["persona"]
feed_collection = db["feed"]
user_collection = db["user"]
vision_models = ['haiku', 'sonnet', 'gpt4v']

httpx_client = httpx_cache.AsyncClient()

async def sort_indexes(arr):
    indexed_arr = list(enumerate(arr))

    sorted_indexed_arr = sorted(indexed_arr, key=lambda x: x[1], reverse=True)

    sorted_indexes = [x[0] for x in sorted_indexed_arr]

    return sorted_indexes


async def num_tokens_from_string(string: str, encoding_name: str = 'cl100k_base') -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

import sys
async def get_retriever(persona, language):
    global persona_knowledge_retriever
    global world_info_retriever
    if f'{persona["_id"]}_pk_{language}' in persona_knowledge_retriever:
        return persona_knowledge_retriever[f'{persona["_id"]}_pk_{language}'], world_info_retriever

    persona_knowledge = eval(f"persona['persona_knowledge_{language}']")

    world_info = teyvat_world_info  # 提瓦特大陆

    world = 'teyvat'
    pk_vectorstore = Chroma(f'{persona["_id"]}_pk_{language}', embeddings,
                            persist_directory=chromadb_persist_directory)
    try:
        if pk_vectorstore._collection.count() == 0:
            print(f'pk_vector store returns zero')
            raise Exception
    except:
        pk_vectorstore = await create_chroma(f'{persona["_id"]}_pk_{language}', persona_knowledge)
        print("create personal knowledge space complete")
    persona_knowledge_retriever = pk_vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 50})

    return persona_knowledge_retriever, world_info_retriever


async def create_chroma(name, text: str):
    obj = hub.pull("wfh/proposal-indexing")
    llm = await create_model("gpt4")
    # use it in a runnable
    runnable = obj | llm

    class Sentences(BaseModel):
        sentences: List[str]

    # Extraction
    extraction_chain = create_extraction_chain_pydantic(pydantic_schema=Sentences, llm=llm)

    async def get_propositions(text):
        runnable_output = (await runnable.ainvoke({
            "input": text
        })).content
        propositions = (await extraction_chain.ainvoke(runnable_output))['text'][0].sentences
        return propositions

    propositions = await get_propositions(text)
    vectorstore = Chroma(name, embeddings, persist_directory=chromadb_persist_directory)
    await vectorstore.aadd_texts(texts=propositions, embedding=embeddings)
    return vectorstore


async def get_memory(chain_dict: dict) -> dict:
    start_time = time.time()
    query = chain_dict['query']
    chat_history = chain_dict['memory'][2:]
    short_term_memory = chat_history[-4:]
    long_term_memory = chat_history[:-4]
    # 给short_term_memory添加query
    short_term_memory.append(HumanMessage(query))

    # 从long term memory中找到最相似的对话

    search_query = await messages_to_string(short_term_memory)  # Search Query可以接着进入LLM综合成一个更好的问题来帮助搜索。
    search_query_embedding = await embeddings.aembed_query(search_query)
    dialogues = []
    if len(long_term_memory) == 0:
        return {'short_term_memory': short_term_memory, 'long_term_memory': long_term_memory,
                'search_query': search_query, **chain_dict}
    for i in range(0, len(long_term_memory), 2):
        dialogues.append(await messages_to_string(long_term_memory[i:i + 2]))
    dialogues_emb = await embeddings.aembed_documents(dialogues)
    similarity = cosine_similarity([search_query_embedding], dialogues_emb)[0]
    most_similar = await sort_indexes(similarity)
    num_tokens = 0
    selected_indexes = []
    for i in most_similar:
        num_tokens += await num_tokens_from_string(dialogues[i])
        if num_tokens > 4096 or similarity[i] < 0.6:
            break
        selected_indexes.append(i)

    # long_term_memory只保留在selected_indexes中的对话
    selected_indexes = sorted(selected_indexes)  # 从小到大排序以保证顺序。
    keep_long_term_memory = []
    for i in selected_indexes:
        keep_long_term_memory.append(long_term_memory[2 * i])
        keep_long_term_memory.append(long_term_memory[2 * i + 1])
    
    print(f'get memory time is {time.time() - start_time}')
    return {'short_term_memory': short_term_memory, 'long_term_memory': keep_long_term_memory,
            'search_query': search_query,
            **chain_dict}


async def remove_image_from_message(messages: list[BaseMessage]) -> list[BaseMessage]:
    for i, mes in enumerate(messages):
        if isinstance(mes.content, list):
            mes.content = mes.content[0]['text']
    return messages

async def remove_name_from_message(messages: list[BaseMessage]) -> list[BaseMessage]:
    for i, mes in enumerate(messages):
        mes.name = None
    return messages


async def get_setting(chain_dict: dict) -> dict:
    start_time = time.time()
    memory = chain_dict['long_term_memory'] + chain_dict['short_term_memory']
    starter = chain_dict['memory'][:2]
    persona = chain_dict['persona']
    search_query = chain_dict['search_query']
    world_info_retriever = chain_dict['world_info_retriever']
    persona_knowledge_retriever = chain_dict['persona_knowledge_retriever']
    language = chain_dict['language']
    world_info_task = asyncio.create_task(world_info_retriever.ainvoke(search_query))
    persona_knowledge_task = asyncio.create_task(persona_knowledge_retriever.ainvoke(search_query))

    world_info, persona_knowledge = await asyncio.gather(world_info_task, persona_knowledge_task)
    print(type(world_info))
    persona_name = await translate_keywords(persona['name'], chain_dict['language'])
    persona_card = eval(f"persona['persona_card_{language}']")

    print(f"get setting check point 1 takes {time.time() - start_time}")
    cur_time = time.time()

    if chain_dict['model_name'] not in vision_models:
        memory = await remove_image_from_message(memory)
        starter = await remove_image_from_message(starter)
        memory = await remove_name_from_message(memory)
        starter = await remove_name_from_message(starter)

    print(f"get setting check point 12takes {time.time() - cur_time}")
    cur_time = time.time()

    dict = {"memory": memory, "persona_card": persona_card,
            "world_info": world_info, "persona_knowledge": persona_knowledge,
            "starter": starter,
            "persona_name": persona_name, "user_name": chain_dict['user_name'], "language": language,
            "model_name": "Claude"}
    print(f'get setting time is {time.time() - start_time}')
    return dict


async def messages_to_string(messages: list[BaseMessage], character_name: str = 'Character') -> str:
    messages_str = ''
    for i, mes in enumerate(messages):
        messages_str += f"{character_name}: {mes.content}\n\n" if i % 2 == 1 else f"User: {mes.content}"
    return messages_str


async def initialize_memory(memory, persona, user_name, user_id, language, feed_id):
    messages = await memory.aget_messages()
    if not messages:
        if feed_id:
            feed_starter_prompt = eval(f"feed_starter_prompt_{language}")
            start_time = time.time()
            # async with httpx_cache.AsyncClient() as client:
            response = await httpx_client.get(f'https://d1bfrmh2m0r73y.cloudfront.net/public/feeds/{feed_id}/1.webp')
            if response.status_code == 200:
                image = base64.b64encode(response.content).decode('utf-8')
                image = f"data:image/webp;base64,{image}"
            else:
                raise ValueError("Failed to get image.")
            print("Time to get image:", time.time() - start_time)
            content = [
                {"type": "text", "text": feed_starter_prompt},
                {"type": "image_url", "image_url": {"url": image}}
            ]
            first_mes = HumanMessage(content)
            first_mes.feed_id = feed_id
            first_mes.name = user_name
            first_mes.id = user_id
            messages.append(first_mes)

            feed = feed_collection.find_one({"_id": feed_id})
            text = eval(f"feed['text_{language}']")
            first_res = AIMessage(text)
            first_res.name = await translate_keywords(persona['name'], language)
            first_res.id = persona['_id']
            await memory.aadd_messages([first_mes, first_res])
            messages.append(first_res)
        else:
            first_mes = HumanMessage('<START-ROLEPLAY>')
            first_mes.name = user_name
            first_mes.id = user_id
            first_mes.feed_id = None
            messages.append(first_mes)
            first_res = AIMessage(eval(f"persona['first_response_{language}']").format(user_name=user_name))
            first_res.name = await translate_keywords(persona['name'], language)
            first_res.id = persona['_id']
            await memory.aadd_messages([first_mes, first_res])
            messages.append(first_res)
    return messages

import traceback

async def dialogue_chain(thread_id, persona, speaker, user_name, model_name, query, user_id, language, feed_id):
    model = await create_model(model_name)
    global testing_time
    print(f"model creating time complete in {time.time() - testing_time}")
    testing_time = time.time()

    async with memory_lock:
        memory.session_id = thread_id
        messages = await initialize_memory(memory, persona, user_name, user_id, language, feed_id)

    print(f"memory initialization complete in {time.time() - testing_time}")
    testing_time = time.time()

    if model is None:
        raise ValueError("Failed to create model. The model is None.")
    
    system_prompt = eval(f"system_prompt_{language}")
    prefix_system_prompt = eval(f"prefix_system_prompt_{language}")
    suffix_system_prompt = eval(f"suffix_system_prompt_{language}")
    persona_name = await translate_keywords(persona['name'], language)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt.format(prefix_system_prompt=prefix_system_prompt,
                                            suffix_system_prompt=suffix_system_prompt)),
            MessagesPlaceholder(variable_name="starter"),
            MessagesPlaceholder(variable_name="memory"),
            ("assistant", f"{persona_name}:"),
        ]
    )
    persona_knowledge_retriever, world_info_retriever = await get_retriever(persona, language)

    print(f"knowledge retrieval complete in {time.time() - testing_time}")
    testing_time = time.time()

    output_parser = StrOutputParser()
    chain = (
            RunnableLambda(get_memory)
            | RunnableLambda(get_setting)
            | prompt
            | model
            | output_parser
    )
    
    chain_dict = {'query': f"{user_name}: "+query, 'memory': messages, 'persona': persona,
                  'persona_knowledge_retriever': persona_knowledge_retriever, 'language': language,
                  'world_info_retriever': world_info_retriever, 'user_name': user_name, 'model_name': model_name}
    # print(f'the test chain is {await prompt_test_chain.ainvoke(chain_dict)}')

    response = ''
    for i in range(3):
        try:
            response = ''
            # response_chain = await chain.astream(chain_dict)
            # print(f'the result chain is {response_chain}')
            async for s in chain.astream(chain_dict):
                response += s
                yield s
            break
        except Exception as e:
            print("An exception occurred:")
            print(traceback.format_exc())
            await asyncio.sleep(4)
            # pass
  
    if response == '':
        raise ValueError("Failed to get response.")

    query = HumanMessage(f"{user_name}: "+query)
    query.name = user_name
    query.id = user_id
    ai_response = AIMessage(f"{persona_name}: "+response)
    ai_response.name = persona_name
    ai_response.id = persona['_id']
    async with memory_lock:
        memory.session_id = thread_id
        await memory.aadd_messages([query, ai_response])
    # 总聊天次数+1
    # persona_collection.update_one({'_id': persona['_id']}, {'$inc': {'chat': 1}})

async def get_feed_user_prompt_with_image(chain_dict: dict) -> dict:
    text = chain_dict['query']['text']
    if text == '':
        text = ' '
    image = chain_dict['query']['image']
    persona_knowledge_retriever = chain_dict['persona_knowledge_retriever']
    world_info_retriever = chain_dict['world_info_retriever']
    world_info = await world_info_retriever.ainvoke(text)
    persona_knowledge = await persona_knowledge_retriever.ainvoke(text)
    persona_card = eval(f"chain_dict['persona']['persona_card_{chain_dict['language']}']")
    if chain_dict['model_name'] == 'yi-vl-plus':
        world_info = ''
        persona_knowledge = ''
        persona_card = ''
    feed_user_prompt = eval(f"feed_user_prompt_{chain_dict['language']}")
    feed_user_prompt_with_image = [HumanMessage(
        content=[
            {"type": "text",
             "text": feed_user_prompt.format(user_name=chain_dict['user_name'], language=chain_dict['language'],
                                             persona_name=chain_dict['persona']['name'], text=text)},
            {
                "type": "image_url",
                "image_url": {
                    "url": image
                },
            },
        ]
    )]

    return {"feed_user_prompt_with_image": feed_user_prompt_with_image, "world_info": world_info,
            "persona_knowledge": persona_knowledge, "persona_name": chain_dict['persona']['name'],
            "user_name": chain_dict['user_name'], "language": chain_dict['language'],
            "persona_card": persona_card, "model_name": "Claude"}


async def feed_completion_chain(persona: dict, user_name: str, model_name: str, query: dict, user_id: str, feed_id: str,
                                language: str, supported_languages: list[str]):
    model = await create_model(model_name)
    system_prompt = eval(f"system_prompt_{language}")
    prefix_system_prompt = eval(f"prefix_system_prompt_{language}")
    suffix_system_prompt = eval(f"suffix_system_prompt_{language}")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt.format(prefix_system_prompt=prefix_system_prompt,
                                            suffix_system_prompt=suffix_system_prompt)),
            MessagesPlaceholder(variable_name="feed_user_prompt_with_image"),
            ("assistant", f"{await translate_keywords(persona['name'], language)}:"),
        ]
    )
    persona_knowledge_retriever, world_info_retriever = await get_retriever(persona, language)
    output_parser = StrOutputParser()
    chain = (
            RunnableLambda(get_feed_user_prompt_with_image)
            | prompt
            | model
            | output_parser
    )
    chain_dict = {'query': query, 'persona': persona, 'persona_knowledge_retriever': persona_knowledge_retriever,
                  'world_info_retriever': world_info_retriever, 'user_name': user_name, 'language': language,
                  'model_name': model_name}
    # print(await prompt_test_chain.ainvoke(chain_dict))
    response = ''
    async for s in chain.astream(chain_dict):
        response += s
        yield s
    memory = MongoDBChatMessageHistory(
        session_id=feed_id,
        connection_string=uri,
        database_name="solaris",
        collection_name="chat_histories",
    )

    for lang in supported_languages:
        if lang == language:
            await memory.aadd_messages([AIMessage(response)])
            print(lang, ":", response)
        else:
            lang_response = await translate_chain(response, lang, "sonnet")
            await memory.aadd_messages([AIMessage(lang_response)])
            print(lang, ":", lang_response)


async def translate_keywords(query: str, to_lang: str):
    filename = f"./i18n/{to_lang}.json"
    try:
        with open(filename, "r", encoding="utf-8") as f:
            lang_dict = json.load(f)
    except FileNotFoundError:
        print(f"找不到 {filename} 文件")
        return query

    # 按照词组长度降序排列
    phrases = sorted(lang_dict.keys(), key=lambda x: len(x), reverse=True)

    # 遍历词组并替换
    for phrase in phrases:
        for value in lang_dict[phrase]:
            if value in query:
                query = query.replace(value, phrase)

    return query


async def translate_chain(query: str, to_lang: str, model_name: str):
    model = await create_model(model_name)
    output_parser = StrOutputParser()
    translate_system_prompt = eval(f"translate_system_prompt_{to_lang}")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", translate_system_prompt),
            ("user", "{query}\n\nOnly returns translations. Please do not explain my original text."),
            ("assistant", "Here is translation in {to_lang}:"),
        ]
    )
    query = await translate_keywords(query, to_lang)

    chain = (
            prompt
            | model
            | output_parser
    )
    return (await chain.ainvoke({'query': query, 'to_lang': to_lang})).strip()
