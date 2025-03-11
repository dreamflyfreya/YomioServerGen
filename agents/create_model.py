from config import *
from langchain_community.chat_models import BedrockChat
from langchain_openai import ChatOpenAI

async def create_model(modelName="gpt4", key=openai_key):
    if modelName == "haiku":
        model = BedrockChat(  # 使用AWS 的anthropic
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            streaming=True,
            model_kwargs={"temperature": 1.0},
            region_name="us-east-1")
    elif modelName == "sonnet":
        model = BedrockChat(  # 使用AWS 的anthropic
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            streaming=True,
            model_kwargs={"temperature": 1.0},
            region_name="us-east-1")
    elif modelName == "gpt4":
        model = ChatOpenAI(model="gpt-4-turbo-preview", openai_api_key=key, temperature=0.6)
    elif modelName == "gpt3.5":
        model = ChatOpenAI(model="gpt-3.5-turbo-0125", openai_api_key=key, temperature=0.6)
    elif modelName == "yi-34b-chat":
        model = ChatOpenAI(model="yi-34b-chat-200k", temperature=0.6,
                           openai_api_key="91df970e9a2249f5a1bcc7204feb004d",
                           openai_api_base="https://api.lingyiwanwu.com/v1", max_tokens=100000)
    elif modelName == "yi-vl-plus":
        model = ChatOpenAI(model="yi-vl-plus", temperature=0.6,
                           openai_api_key="91df970e9a2249f5a1bcc7204feb004d",
                           openai_api_base="https://api.lingyiwanwu.com/v1")
    elif modelName == "gpt4v":
        model = ChatOpenAI(model="gpt-4-vision-preview", temperature=1.0, openai_api_key=key)
    elif modelName == "mixtral":
        model = BedrockChat(  # 使用AWS 的anthropic
            model_id="mistral.mixtral-8x7b-instruct-v0:1",
            streaming=True,
            model_kwargs={"temperature": 0.6},
            region_name="us-east-1")
    else:
        model = None
    return model
