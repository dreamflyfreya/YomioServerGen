from urllib.parse import urlencode
import requests

# 基础 URL
base_url = "http://54.185.44.246:5000/voice"


def fetch_audio(speaker, text, language="ZH", length=1, noise=0.6, noisew=0.9, sdp_ratio=0.5, emotion="Happy"):
    request_json = {
        'auto_split': 'false',
        'auto_translate': 'false',
        'emotion': emotion,
        'language': language,
        'length': length,
        'model_id': '0',
        'noise': noise,
        'noisew': noisew,
        'sdp_ratio': sdp_ratio,
        'speaker_name': speaker,
        'style_weight': '0',
        'text': text}

    # 编码查询参数  
    encoded_params = urlencode(request_json)

    # 构造完整 URL
    full_url = f"{base_url}?{encoded_params}"

    # 发送 GET 请求
    response = requests.get(full_url)

    # 检查响应状态码
    if response.status_code == 200:
        # 返回响应的字节流
        return response.content
    else:
        print(f"Request failed with status code {response.status_code}")
        return None


if __name__ == "__main__":
    res = fetch_audio({'八重神子_ZH': 1}, "The atmosphere of this tavern is warm and inviting, with a hint of mystery that lingers in the air. The lanterns cast a soft glow, creating a cozy ambiance that welcomes all who step inside. The sound of laughter and chatter fills the room, mingling with the aroma of hearty food and fine drinks.")
    with open('output.wav', 'wb') as f:
        f.write(res)
