from abc import ABC, abstractmethod
from asyncio import Event

from fastapi import WebSocket


class TextToSpeech(ABC):
    @abstractmethod
    async def stream(
        self,
        text: str,
        websocket: WebSocket,
        tts_event: Event,
        voice_id: str,
        first_sentence: bool,
        language: str,
        *args,
        **kwargs
    ):
        pass

    async def generate_audio(self, *args, **kwargs):
        pass
