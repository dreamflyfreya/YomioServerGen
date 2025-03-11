from langchain_community.vectorstores.chroma import Chroma
import os
from langchain import hub
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import List
from .create_model import create_model

from langchain import hub
from langchain_community.chat_models import BedrockChat
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_extraction_chain_pydantic
from langchain_core.pydantic_v1 import BaseModel
from langchain import hub
from .prompts import *
import asyncio
from config import *
from pymongo.mongo_client import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient

chromadb_persist_directory = os.path.expanduser('~/chromadb')
embeddings = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=openai_key)

from agents.logger import *
logger = get_logger(__name__)

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

async def get_personal_retriever(persona, language):
    persona_knowledge = eval(f"persona['persona_knowledge_{language}']")

    pk_vectorstore = Chroma(f'{persona["_id"]}_pk_{language}', embeddings,
                            persist_directory=chromadb_persist_directory)
    try:
        if pk_vectorstore._collection.count() == 0:
            logger.info(f'pk_vector store returns zero')
            raise Exception
    except:
        pk_vectorstore = await create_chroma(f'{persona["_id"]}_pk_{language}', persona_knowledge)
        logger.info("create personal knowledge space complete")
    persona_knowledge_retriever = pk_vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 50})
    return f'{persona["_id"]}_pk_{language}', persona_knowledge_retriever


async def get_world_retriever(language):
    world_info = teyvat_world_info  # 提瓦特大陆

    world = 'teyvat'

    wi_vectorstore = Chroma(f'{world}_wi_{language}', embeddings, persist_directory=chromadb_persist_directory)
    try:
        if wi_vectorstore._collection.count() == 0:
            raise Exception
    except:
        logger.info("creating world vi store")
        wi_vectorstore = await create_chroma(f'{world}_wi_{language}', world_info)
    logger.info("knowledge base created")
    world_info_retriever = wi_vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 50})
    return world_info_retriever

# to do optimize the 
def initialize_knowledge_retriever(persona_name):
    client = AsyncIOMotorClient(f"mongodb://{mongo_uri}:{mongo_port}")

    db = client.solaris
    persona_collection = db["persona"]
    persona_knowledge_retrivers = {}
    async def persona_async_generator():
        async for value in persona_collection.find({"name": {"$in": persona_name}}):
            persona_id, persona_knowledge_retriever = await get_personal_retriever(value, "zh")
            persona_knowledge_retrivers[persona_id] = persona_knowledge_retriever
    loop = asyncio.get_event_loop()
    loop.run_until_complete(persona_async_generator())
    world_knowledge_retrivers = loop.run_until_complete(get_world_retriever("zh"))
    return persona_knowledge_retrivers, world_knowledge_retrivers

