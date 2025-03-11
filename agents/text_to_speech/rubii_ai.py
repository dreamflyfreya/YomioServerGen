import asyncio
import base64
import os
import types

import httpx
import text_to_speech
base_url = "http://tts.autogame.ai/voice"

from agents.logger import *
logger = get_logger(__name__)

class RubiiAiTTS(text_to_speech):
    def __init__(self):
        super().__init__()

    async def stream(
        self,
        text,
        voice_id="21m00Tcm4TlvDq8ikWAM",
        language="ZH",
        length=1, 
        noise=0.6, 
        noisew=0.9, 
        sdp_ratio=0.5, 
        emotion="Happy"
    ):
        if voice_id == "":
            voice_id = "21m00Tcm4TlvDq8ikWAM"
        # headers = config.headers
        # if language != "en-US":
        #     config.data["model_id"] = ELEVEN_LABS_MULTILINGUAL_MODEL
        # data = {
        #     "text": text,
        #     **config.data,
        # }
        # url = config.url.format(voice_id=voice_id)
        # url += "?output_format=" + ("ulaw_8000" if platform == "twilio" else "mp3_44100_128")
        request_json = {
            'auto_split': 'false',
            'auto_translate': 'false',
            'emotion': emotion,
            'language': language,
            'length': length,
            'noise': noise,
            'noisew': noisew,
            'sdp_ratio': sdp_ratio,
            'speaker_name': {"艾尔海森": 1},
            'style_weight': '0',
            'text': text
        }
        async with httpx.AsyncClient() as client:
            response = await client.get(base_url, json=request_json)
            if response.status_code != 200:
                print(f"ElevenLabs returns response {response.status_code}")
            async for chunk in response.aiter_bytes():
                await asyncio.sleep(0.1)
                yield chunk

    # async def generate_audio(self, text, voice_id="", language="en-US") -> bytes:
    #     if voice_id == "":
    #         logger.info("voice_id is not found in .env file, using ElevenLabs default voice")
    #         voice_id = "21m00Tcm4TlvDq8ikWAM"
    #     headers = config.headers
    #     if language != "en-US":
    #         config.data["model_id"] = ELEVEN_LABS_MULTILINGUAL_MODEL
    #     data = {
    #         "text": text,
    #         **config.data,
    #     }
    #     # Change to non-streaming endpoint
    #     url = config.url.format(voice_id=voice_id).replace("/stream", "")
    #     async with httpx.AsyncClient() as client:
    #         response = await client.post(url, json=data, headers=headers)
    #         if response.status_code != 200:
    #             logger.error(f"ElevenLabs returns response {response.status_code}")
    #         # Get audio/mpeg from the response and return it
    #         return response.content
