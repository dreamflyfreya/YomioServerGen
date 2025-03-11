from agents.chains import translate_chain
import json
import uuid

def split_and_combine(string, max_chars):
    # 按照\n分段
    segments = string.split('\n')
    
    result = []
    current_segment = ''
    
    for segment in segments:
        # 如果当前段落加上新的段落长度不超过最大字符数量
        if len(current_segment) + len(segment) <= max_chars:
            current_segment += segment
        else:
            # 如果超过最大字符数量,将当前段落加入结果列表
            result.append(current_segment)
            current_segment = segment
    
    # 将最后一个段落加入结果列表
    if current_segment:
        result.append(current_segment)
    
    return result 

async def create_persona(body):
    creator_id = body.get("creator_id")
    character_name = body.get("speaker")

    persona_card = body.get['persona_card_zh']
    persona_knowledge = body.get['persona_knowledge_zh']
    first_response = body.get['first_response_zh']

    persona_card_en = ''
    
    for string in split_and_combine(persona_card, 900):
        persona_card_en += (await translate_chain(string, 'en', 'gpt4')).strip()
    persona_knowledge_en = ''

    for string in split_and_combine(persona_knowledge, 900):
        persona_knowledge_en += (await translate_chain(string, 'en', 'gpt4')).strip()

    first_response_en = (await translate_chain(first_response, 'en', 'gpt4')).strip()

    persona = {
        "_id": str(uuid.uuid1()),
        "name": character_name,
        "creator": creator_id,
        "contractor": "",
        "shareholders": [
            ""
        ],
        "worlds": [],
        "prefix_system_prompt": "",
        "suffix_system_prompt": "",
        "feed_prefix_system_prompt": "",
        "feed_suffix_system_prompt": "",
        "stream_prefix_system_prompt": "",
        "stream_suffix_system_prompt": "",
        "sample_dialogues": [],
        "voice_params": {
            "speaker": {
            f"{character_name}": 1.0, 
            },
            "sdp_ratio": 0.8,
            "noise": 0.9,
            "noise_W": 0.9,
            "length": 1.0,
        },
        "rank": 1,
        "feed_ids": [
        
        ],
        "chat": 0,
        "follower": 0,
        "persona_card_en": "",
        "first_response_en": "",
        "introduction_en": "",
        "first_response_zh": "",
        "introduction_zh": "",
        "persona_card_zh": "",
        "persona_knowledge_zh": "",
    }

    persona['persona_card_en'] = persona_card_en
    persona['persona_knowledge_en'] = persona_knowledge_en
    persona['persona_card_zh'] = persona_card
    persona['persona_knowledge_zh'] = persona_knowledge
    persona['first_response_zh'] = first_response
    persona['first_response_en'] = first_response_en
    persona['introduction_zh'] = first_response
    persona['introduction_en'] = first_response_en

    return persona
