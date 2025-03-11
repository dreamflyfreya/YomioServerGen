import emoji
import re
from langchain.callbacks.base import AsyncCallbackHandler
from ..text_to_speech import TextToSpeech

from agents.logger import *
logger = get_logger(__name__)

class AsyncCallbackAudioHandler(AsyncCallbackHandler):
    def __init__(
        self,
        voice_id: str = "",
        text_to_speech: TextToSpeech = None,
        language: str = "en-US",
        sid: str = "",
        platform: str = "",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.current_sentence = ""
        self.voice_id = voice_id
        self.language = language
        # self.is_reply = False  # the start of the reply. i.e. the substring after '>'
        # self.twilio_stream_id = sid
        self.platform = platform
        self.text_to_speech = text_to_speech
        # optimization: trade off between latency and quality for the first sentence
        self.sentence_idx = 0

    async def on_chat_model_start(self, *args, **kwargs):
        pass

    async def on_llm_new_token(self, token: str, *args, **kwargs):
        # skip emojis
        token = emoji.replace_emoji(token, "")
        token = self.text_regulator(token)
        if not token:
            return
        for char in token:
            await self._on_llm_new_character(char)

    async def _on_llm_new_character(self, char: str):
        # send to TTS in sentences
        punctuation = False
        if (
            # English punctuations
            (
                char == " "
                and self.current_sentence != ""
                and self.current_sentence[-1] in {".", "?", "!"}
            )
            # Chinese/Japanese/Korean punctuations
            or (char in {"。", "？", "！"})
            # newline
            or (char in {"\n", "\r", "\t"})
        ):
            punctuation = True

        self.current_sentence += char

        if self.text_to_speech == None:
            logger.info("text to speech is none")

        if punctuation and self.current_sentence.strip():
            # first_sentence = self.sentence_idx == 0
            # if first_sentence:
            #     timer.log("LLM First Sentence", lambda: timer.start("TTS First Sentence"))
            await self.text_to_speech.stream(
                text=self.current_sentence.strip(),
                voice_id=self.voice_id,
                language=self.language,
            )
            self.current_sentence = ""
            # timer.log("TTS First Sentence")
            self.sentence_idx += 1

    async def on_llm_end(self, *args, **kwargs):
        first_sentence = self.sentence_idx == 0
        if self.current_sentence.strip():
            await self.text_to_speech.stream(
                text=self.current_sentence.strip(),
                websocket=self.websocket,
                tts_event=self.tts_event,
                voice_id=self.voice_id,
                first_sentence=first_sentence,
                language=self.language,
                priority=self.sentence_idx,
            )

    def text_regulator(self, text):
        pattern = (
            r"[\u200B\u200C\u200D\u200E\u200F\uFEFF\u00AD\u2060\uFFFC\uFFFD]"  # Format characters
            r"|[\uFE00-\uFE0F]"  # Variation selectors
            r"|[\uE000-\uF8FF]"  # Private use area
            r"|[\uFFF0-\uFFFF]"  # Specials
        )
        filtered_text = re.sub(pattern, "", text)
        return filtered_text