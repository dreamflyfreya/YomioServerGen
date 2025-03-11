import asyncio
import json
import time
import uuid
import io
import requests
from fastapi import (
    FastAPI,
    Body,
    Header,
    HTTPException,
    Depends,
    Request,
    status,
    Response,
    BackgroundTasks
)
from openai import AsyncOpenAI
import random
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from langchain_mongodb import MongoDBChatMessageHistory
from passlib.context import CryptContext
from typing import List
from pydantic import BaseModel
import jwt
from datetime import datetime, timedelta
from helpers import (
    get_preferred_language
)
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from agents.chains import *
from fetch_audio import fetch_audio
from create_persona import create_persona
# uri = "mongodb+srv://linkangzhan:Junity069210@solaris.58yruig.mongodb.net/"
# Create a new client and connect to the server

from config import *

client = MongoClient(mongo_uri, int(mongo_port))
uri = f"mongodb://{mongo_uri}:{mongo_port}"

try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)
db = client.solaris

persona_collection = db["persona"]

persona_collection = db["persona"]
feed_collection = db["feed"]
user_collection = db["user"]
local_personas = list(persona_collection.find())

supported_languages = ["en", "zh"]

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class CharacterData(BaseModel):
    name: str
    followers: int
    chat: int
    rank: int
    message: str
    supportedPeople: List[str]
    feed: List[List[str]]


class CommentData(BaseModel):
    comment_name: str
    comment: str
    comment_like: int
    comment_dislike: int
    date: str


class FeedData(BaseModel):
    id: str
    character: str
    images: List[str]
    text: str
    date: str
    like: int
    dislike: int
    comments_list: List[CommentData]


SECRET_KEY = "asdfghjkl"  # 更换为你的秘钥
ALGORITHM = "HS256"  # JWT使用的算法
ACCESS_TOKEN_EXPIRE_MINUTES = 10080


def hash_password(password: str):
    return pwd_context.hash(password)


def authenticate_user(email: str, password: str) -> bool | dict:
    user = user_collection.find_one({"email": email})
    if not user:
        return False
    if not pwd_context.verify(password, user['password']):
        return False
    return user


def get_user_id_from_token(x_token: str = Header(...)):
    try:
        payload = jwt.decode(x_token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="User ID not found in token")
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


def create_access_token(data: dict, expires_delta: timedelta = 60):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 允许访问的源列表
    allow_credentials=True,
    allow_methods=["*"],  # 允许的方法
    allow_headers=["*"],  # 允许的头部
)


@app.post("/api/login")
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    print(form_data.username, form_data.password)
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        # raise HTTPException(
        #     status_code=status.HTTP_401_UNAUTHORIZED,
        #     detail="Incorrect email or password",
        #     headers={"WWW-Authenticate": "Bearer"},
        # )
        return {"error": "Incorrect email or password"}
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["_id"]}, expires_delta=access_token_expires
    )
    user.pop("password")
    print(f'the access token is {access_token}')
    return {"access_token": access_token, "token_type": "bearer", "user": json.dumps(user, ensure_ascii=False),
            "message": "Login successful"}


@app.post("/api/register")
def register(request: Request, name: str = Body(...), password: str = Body(...), email: str = Body(...)):
    if user_collection.find_one({"email": email}):
        return {"error": "Email already registered"}
    hashed_password = hash_password(password)
    user_collection.insert_one({
        "_id": str(uuid.uuid1()),
        "name": name,
        "password": hashed_password,
        "email": email,
        "token": 100,
        "items": {
        },
        "threads": {

        }
    })
    return {"message": "User successfully registered", "name": name}


@app.get("/api/get-feeds")
async def get_character_info(request: Request, feed_ids: str, user_id: str = Depends(get_user_id_from_token)):
    language = get_preferred_language(request.headers.get("Accept-Language", "en"), supported_languages)
    feed_ids = feed_ids.split(',')
    feeds = list(feed_collection.find({"_id": {"$in": feed_ids}}))
    random.shuffle(feeds)
    for feed in feeds:
        feed['text'] = feed.get('text_' + language, feed.get('text', ''))
    return feeds


@app.get("/api/get-persona")
async def get_persona(request: Request, persona_id: str = '', fields: str = '',
                      user_id: str = Depends(get_user_id_from_token)):
    language = get_preferred_language(request.headers.get("Accept-Language", "en"), supported_languages)
    fields = fields.split(',')
    # persona = persona_collection.find_one({"_id": persona_id})
    persona = next((persona for persona in local_personas if persona["_id"] == persona_id), None).copy()
    persona['name'] = await translate_keywords(persona['name'], language),
    persona['text'] = persona.get('text_' + language, persona.get('text', ''))
    persona['introduction'] = persona.get('introduction_' + language, persona.get('introduction', ''))
    persona['first_response'] = persona.get('first_response_' + language, persona.get('first_response', ''))

    if not persona:
        raise (HTTPException(status_code=404, detail="Persona not found"))
    return {field: persona.get(field, '') for field in fields}


@app.get("/api/get-user")
async def get_user(request: Request, user_id: str = Depends(get_user_id_from_token)):
    return user_collection.find_one({"_id": user_id})


@app.post("/api/set-name")
async def set_name(request: Request, name: str, user_id: str = Depends(get_user_id_from_token)):
    language = get_preferred_language(request.headers.get("Accept-Language", "en"), supported_languages)
    user_collection.update_one({"_id": user_id}, {"$set": {"name": name}})
    return {"message": "Name updated"}

@app.get("/api/get-id")
async def get_id(request: Request, name='', user_id: str = Depends(get_user_id_from_token)):
    language = get_preferred_language(request.headers.get("Accept-Language", "en"), supported_languages)
    if name == '':
        return user_id
    # return str(persona_collection.find_one({"name": name})['_id'])
    return str(next((persona["_id"] for persona in local_personas if persona["name"] == name), None))

@app.get("/api/feeds")
async def get_feeds(request: Request, user_id: str = Depends(get_user_id_from_token)):
    start_time = time.time()
    feeds = []
    language = get_preferred_language(request.headers.get("Accept-Language", "en"), supported_languages)
    for feed in feed_collection.find():
        index = "text_" + language
        persona = next((persona for persona in local_personas if persona["_id"] == feed["persona_id"]), None)
        feed_data = {
            "_id": feed["_id"],
            "text": feed[index],
            "persona_id": persona["_id"],
            "name": await translate_keywords(persona["name"], language),
            "follower": persona["follower"],
        }
        feeds.append(feed_data)
    random.shuffle(feeds)
    return feeds


@app.post("/api/delete-last-message")
async def delete_last_message(request: Request, user_id: str = Depends(get_user_id_from_token)):
    body = await request.json()
    thread_id = body.get("thread_id")
    memory = MongoDBChatMessageHistory(
        session_id=thread_id,
        connection_string=uri,
        database_name="solaris",
        collection_name="chat_histories",
    )
    messages = await memory.aget_messages()
    await memory.aclear()
    await memory.aadd_messages(messages[:-2])
    return {"success": "success"}

async def audio_transcribe(audio_data):
    audio_buffer = io.BytesIO()
    api_key = openai_key
    openai_client = AsyncOpenAI(api_key=api_key)
    # Read the contents of the UploadFile object
    audio_bytes = await audio_data.read()

    # Write the bytes to the audio buffer
    audio_buffer.write(audio_bytes)

    print("aduio buffer written")
    audio_buffer.name = "file.mp3"
    transcription = await openai_client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_buffer,
        response_format="text"
    )
    return transcription

import re
def response_parse(sentence):
    double_asterisk_words = re.findall(r'\*\*(.*?)\*\*', sentence)
    single_asterisk_words = re.findall(r'\*(.*?)\*', sentence)
    
    return double_asterisk_words, single_asterisk_words

# audio transmitted in bytes
import io
@app.route("/api/send-message", methods=['POST'])
async def send_message(request: Request, x_token: str = Header(...)):
    body = await request.form()
    headers = request.headers
    try:
        x_token = headers.get('x-token')
    except:
        raise HTTPException(status_code=401, detail="Missing x-token header")
    user_id = get_user_id_from_token(x_token)
    language = body.get("language")
    persona_id = body.get("persona_id")
    speaker = body.get("speaker")
    persona = persona_collection.find_one({"_id": persona_id})
    if persona is None:
        raise HTTPException(status_code=404, detail=f"persona not found for {persona_id}")
    # persona = next((persona for persona in local_personas if persona["_id"] == persona_id), None).copy()
    thread_id = body.get("thread_id")
    feed_id = body.get("feed_id")
    nsfw = body.get("nsfw")
    user_name = user_collection.find_one({"_id": user_id}).get("name", user_id)
    model_name = body.get("model_name", "haiku")
    global testing_time
    testing_time = time.time()

    query = await audio_transcribe(body.get('audio_data'))
    print(f"audio transcribing finishes in {time.time() - testing_time}")
    testing_time = time.time()

    query = "你叫什么名字"
    print(f"transcribed query is {query}")
    # if nsfw:
    #     print('nsfw')
    # if query.get('audio', None):
    #     pass  # TODO: 语音识别
    # if query.get('image', None):
    #     pass  # TODO: 图像识别
    
    # full_response = ""
    # async for response in dialogue_chain(thread_id=thread_id, persona=persona, speaker=speaker, user_name=user_name, model_name=model_name,
    #     query=query, user_id=user_id, language=language, feed_id=feed_id):
    #     #print(f"response got is {response}")
    #     full_response += response
    
    # print(f"message fetching complete in {time.time() - testing_time}")
    # testing_time = time.time()

    # full_words, _ = response_parse(full_response)
    # print(f"the full words is {full_words[0]}")
    
    # audio_content = await fetch_audio(speaker={speaker: 1}, text=full_words[0])
    # with open("fetched_audio.wav", "wb") as file:
    #     file.write(audio_content)
    # print(f"message transcribe complete in {time.time() - testing_time}")
    # testing_time = time.time()

    async def audio_chunk_generator(dialogue_chain, speaker, language):
        sentence = ""
        testing_time = time.time()
        valid_word = False
        async for word in dialogue_chain:
            if re.search(r'[^\w\s]', word):
                # Generate audio for the complete sentence
                #print(sentence)
                if not valid_word:
                    continue
                valid_word = False
                print(f'time to generate text is {time.time() - testing_time}')
                print(sentence)
                testing_time = time.time()
                audio_content = await fetch_audio(speaker={speaker: 1}, text=sentence)
                yield audio_content
                sentence = ""  # Reset the sentence for the next iteration
            else:
                valid_word = True
                sentence += word

    testing_time = time.time()

    dialogue_chain_instance = dialogue_chain(thread_id=thread_id, persona=persona, speaker=speaker, user_name=user_name, model_name=model_name, query=query, user_id=user_id, language=language, feed_id=feed_id)

    async def audio_generator():
        async for audio_chunk in audio_chunk_generator(dialogue_chain_instance, speaker, language):
            yield audio_chunk

    return StreamingResponse(
        content=audio_generator(),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment;filename=audio.wav"}
    )


@app.post("/api/feed-completion")
async def feed_completion(request: Request, user_id: str = Depends(get_user_id_from_token)):
    body = await request.json()
    query = body.get("query")
    persona_id = body.get("persona_id")
    # persona = persona_collection.find_one({"_id": persona_id})
    persona = next((persona for persona in local_personas if persona["_id"] == persona_id), None).copy()
    feed_id = body.get("feed_id")
    print("feed id", feed_id)
    language = get_preferred_language(request.headers.get("Accept-Language", "en"), supported_languages)
    if user_id not in persona["shareholders"]:
        raise HTTPException(status_code=400, detail="Bad Request")
    user_name = body.get("user_name", user_id)
    if 'text' not in query and 'image' not in query:
        raise HTTPException(status_code=400, detail="Bad Request")
    return StreamingResponse(feed_completion_chain(persona=persona, user_name=user_name, model_name='sonnet',
                                                   query=query, user_id=user_id, feed_id=feed_id, language=language,
                                                   supported_languages=supported_languages), media_type="text/plain")


@app.post("/api/add-feed")
async def feed_completion(request: Request, background_tasks: BackgroundTasks,
                          user_id: str = Depends(get_user_id_from_token)):
    body = await request.json()
    persona_id = body.get("persona_id")
    feed_id = body.get("feed_id")
    print("feed id", feed_id)
    # persona = persona_collection.find_one({"_id": persona_id})
    persona = next((persona for persona in local_personas if persona["_id"] == persona_id), None).copy()
    if user_id not in persona["shareholders"]:
        raise HTTPException(status_code=400, detail="Bad Request")
    # 毫秒时间戳
    date = int(time.time() * 1000)
    memory = MongoDBChatMessageHistory(
        session_id=feed_id,
        connection_string=uri,
        database_name="solaris",
        collection_name="chat_histories",
    )
    if not await memory.aget_messages():
        raise HTTPException(status_code=400, detail="Bad Request")

    # 在后台运行任务
    background_tasks.add_task(wait_and_update_feed, feed_id, persona_id, date, memory)

    return {"success": "success"}


async def wait_and_update_feed(feed_id, persona_id, date, memory):
    while len(await memory.aget_messages()) < len(supported_languages):
        print(len(await memory.aget_messages()))
        print(await memory.aget_messages())
        print("waiting for other language completion")
        await asyncio.sleep(5)

    feed_collection.insert_one({
        "_id": feed_id,
        "persona_id": persona_id,
        "image": 1,
        "like": 0,
        "dislike": 0,
        "date": date,
        "comments": []
    })
    for i, language in enumerate(supported_languages):
        feed_collection.update_one({"_id": feed_id},
                                   {"$set": {f"text_{language}": (await memory.aget_messages())[i].content}})

    # persona collection的feed_ids字段添加feed_id
    persona_collection.update_one({"_id": persona_id}, {"$push": {"feed_ids": feed_id}})

@app.post("/api/get-thread")
async def get_thread(request: Request, user_id: str = Depends(get_user_id_from_token)):
    print('get thread')

    body = await request.json()
    language = get_preferred_language(request.headers.get("Accept-Language", "en"), supported_languages)
    persona_id = body.get("persona_id")
    new_thread = body.get("new_thread", False)
    feed_id = body.get("feed_id")
    user = user_collection.find_one({"_id": user_id})
    if not user.get("threads", {}).get(persona_id):
        thread_id = str(uuid.uuid1())
        user_collection.update_one(
            {"_id": user_id},
            {"$set": {f"threads.{persona_id}": [thread_id]}})
        global local_personas
        local_personas = list(persona_collection.find())
    elif new_thread:
        thread_id = user["threads"][persona_id][-1]
        memory = MongoDBChatMessageHistory(
            session_id=thread_id,
            connection_string=uri,
            database_name="solaris",
            collection_name="chat_histories",
        )
        await memory.aclear()
    else:
        thread_id = user["threads"][persona_id][-1]
    memory = MongoDBChatMessageHistory(
        session_id=thread_id,
        connection_string=uri,
        database_name="solaris",
        collection_name="chat_histories",
    )
    persona = next((persona for persona in local_personas if persona["_id"] == persona_id), None).copy()
    user_name = user.get("name", user_id)
    messages = await initialize_memory(memory, persona, user_name, user_id, language, feed_id)
    message = messages[0].dict()
    if feed_id:
        if message.get('feed_id', None) != feed_id:
            await memory.aclear()
            messages = await initialize_memory(memory, persona, user_name, user_id, language, feed_id)
    if isinstance(messages[0].content, list):
        messages[1].content = [
            {"type": "text", "text": messages[1].content},
            {"type": "image_url", "image_url": {"url": messages[0].content[1]['image_url']['url']}}
        ]
    return {"thread_id": thread_id, "messages": messages[1:]}


@app.post("/api/create-character")
async def create_character(request: Request, user_id: str = Depends(get_user_id_from_token)):
    body = request.json()

    persona = await create_persona(body)
    persona_collection = db["persona"]

    persona_collection.insert_one(persona)
    


@app.get("/api/get-all-personas")
async def get_all_personas(request: Request, user_id: str = Depends(get_user_id_from_token)):
    language = get_preferred_language(request.headers.get("Accept-Language", "en"), supported_languages)
    personas = []
    # for persona in persona_collection.find():
    for persona in local_personas:
        persona_info = {
            "persona_id": persona["_id"],
            "introduction": persona.get("introduction_" + language, persona.get("introduction", "")),
            "name": await translate_keywords(persona['name'], language),
            "follower": persona["follower"],
            "chat": persona["chat"],
            "rank": persona["rank"]
        }
        personas.append(persona_info)
    return personas


if __name__ == "__main__":
    uvicorn.run("get_message:app", host='0.0.0.0', port=9985, reload=True)
