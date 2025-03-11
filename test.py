import json
import uuid
from agents.chains import *
import json
import requests
from config import *

def create_character():
    os.environ['OPENAI_API_KEY'] = openai_key
    os.environ['openai_api_key'] = openai_key

    print(1)
    character_name = "胡桃"

    with open(f'./persona.json', 'r') as f:
        character = json.load(f)
    print(2)    

    # Create a new client and connect to the server
    client =MongoClient("localhost", 27017)

    db = client.solaris
    print(3)   
    persona_collection = db["persona"]
    feed_collection = db["feed"]
    supported_languages = ['zh', 'en']
    persona_id = str(uuid.uuid1())
    print(4)
    persona_collection.insert_one(character)


def register_and_login():
    api_url = "http://localhost:61001/api/register"  # Replace with your API URL

    payload = {
        "name": "瑞杰",
        "password": "password123",
        "email": "john.doe@example.com"
    }
    register_response = requests.post(api_url, json=payload)

    print(f"register result: {register_response.json}")

    api_url = "http://localhost:61001/api/login"  # Replace with your API URL

    payload = {
        "password": "password123",
        "username": "john.doe@example.com"
    }
    login_response = requests.post(api_url, data=payload)

    return login_response

def get_thread(login_response):
    api_url = "http://localhost:61001/api/get-thread"
    payload = {
        "persona_id": "d3360bf4-efc0-11ee-9fec-7a36fc54fefe",
        "new_thread": True,
        "feed_id": None
    }
    headers = {
        "x-token": f"{login_response.json().get('access_token')}",
        "Accept-Language": "en"
    }
    get_thread_response = requests.post(api_url, json=payload, headers=headers)

    if get_thread_response.status_code == 200:
        return get_thread_response
    else:
        print(f"Get thread failed with status code: {get_thread_response.status_code}")
        print(f"Response content: {get_thread_response.text}")
        return None
    
import base64

def get_audio(login_response, get_thread_response):
    api_url = "http://localhost:61001/api/send-message"
    with open("resulting.wav", "rb") as file:
        payload = {
            "language": "zh",
            "persona_id": "d3360bf4-efc0-11ee-9fec-7a36fc54fefe",
            "speaker": "胡桃_ZH",
            "username": "john.doe@example.com",
            "model_name": "gpt4",
            "thread_id": get_thread_response.json().get("thread_id"),
            #"feed_id": None,
        }

        files = {"audio_data": file}
        headers = {
            "x-token": f"{login_response.json().get('access_token')}",
            #         "language": "ZH",
            # "persona_id": "d3360bf4-efc0-11ee-9fec-7a36fc54fefe",
            # "speaker": "八重神子_ZH",
            # "username": "john.doe@example.com",
            # "model": "gpt4",
            # "thread_id": get_thread_response.json().get("thread_id"),
            #"feed_id": None,
        }
    # json_payload = json.dumps(payload, ensure_ascii=False).encode('utf-8')

        with requests.post(api_url, data=payload, files=files, headers=headers, stream=True) as response:
            with open("received_audio.wav", "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)

# create_character()
login_response = register_and_login()
get_thread_response = get_thread(login_response)
print(get_thread_response.json().get("thread_id"))
print(login_response.json().get('access_token'))

get_audio(login_response, get_thread_response)