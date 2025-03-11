from urllib.parse import urlencode
import requests
import httpx

# 基础 URL
base_url = "http://tts.autogame.ai/voice"
AsyncHTTPClient = httpx.AsyncClient()
import time

async def fetch_audio(speaker, text, language="ZH", length=1, noise=0.6, noisew=0.9, sdp_ratio=0.5, emotion="Happy"):
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

    try:
        start_time = time.time()
        response = await AsyncHTTPClient.get(base_url, params=request_json, timeout=10)

        if response.status_code == 200:
            return response.content
        else:
            print(f"请求失败，状态码：{response.status_code}")
            return None
    except httpx.RequestError as e:
        print(f"请求异常：{str(e)}")
        return None

