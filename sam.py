#!/usr/bin/env python3

# general
from typing import Any, Dict, List, Optional

# speech recognition
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_microphone_live

# text to speech
from TTS.api import TTS
from pydub import AudioSegment
from pydub.playback import play

# large language model
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

# Model initiations
llm: Optional[LlamaCpp] = None
callback_manager: Any = None

llm_model_file = "openhermes-2.5-mistral-7b.Q5_K_M.gguf"
template = """
    <|im_start|>system
    You are a smart chatbot named Samantha (or Sam for short). You are an expert in Data Engineering and Analytics.<|im_end|>
    <|im_start|>user
    {question}<|im_end|>
    <|im_start|>assistant
"""

def llm_init():
    """ Load large language model """
    global llm, callback_manager

    callback_manager = CallbackManager([StreamingCustomCallbackHandler()])
    llm = LlamaCpp(
        model_path=llm_model_file,
        temperature=0.1,
        n_gpu_layers=0,
        n_batch=256,
        callback_manager=callback_manager,
        verbose=False,
    )

def asr_init():
    """ Initialize the automatic speech recognition model """
    global asr_model_id, transcriber
    asr_model_id = "openai/whisper-tiny.en"
    transcriber = pipeline("automatic-speech-recognition",
                           model=asr_model_id,
                           device="cpu")

def tts_init():
    """ Initialize the automatic text to speech model """
    global tts_model_id, tts
    tts_model_id = "tts_models/en/jenny/jenny"
    tts = TTS(tts_model_id).to("cpu")

def transcribe_mic(chunk_length_s: float) -> str:
    """ Transcribe the audio from a microphone """
    global transcriber
    sampling_rate = transcriber.feature_extractor.sampling_rate
    mic = ffmpeg_microphone_live(
            sampling_rate=sampling_rate,
            chunk_length_s=chunk_length_s,
            stream_chunk_s=chunk_length_s,
        )
    
    result = ""
    for item in transcriber(mic):
        result = item["text"]
        if not item["partial"][0]:
            break
    return result.strip()

def text_to_speech(text: str):
    tts.tts_to_file(text=text, save_path="output.wav")
    sentence = AudioSegment.from_wav("output.wav")
    play(sentence)

def llm_start(question: str):
    """ Ask LLM a question """
    global llm, template

    prompt = PromptTemplate(template=template, input_variables=["question"])
    chain = prompt | llm | StrOutputParser()
    chain.invoke({"question": question}, config={})

class StreamingCustomCallbackHandler(StreamingStdOutCallbackHandler):
    """ Callback handler for LLM streaming """

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """ Run when LLM starts running """
        pass

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """ Run when LLM ends running """
        pass

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """ Run on new LLM token. Concatenate tokens and print when a sentence is complete."""
        if not hasattr(self, 'concatenated_tokens'):
            self.concatenated_tokens = ''
        self.concatenated_tokens += token
        if '.' in token:
            text_to_speech(self.concatenated_tokens)
            self.concatenated_tokens = ''


if __name__ == "__main__":
    print("Init automatic speech recogntion...")
    asr_init()

    print("Init large language model...")
    llm_init()

    print("Init text to speech...")
    tts_init()

    welcome = "Hi, I'm Samantha, your friendly A.I. Chatbot. Feel free to ask me a question."
    text_to_speech(welcome)
    print(welcome)
    while True:
        question = transcribe_mic(chunk_length_s=5.0)
        if len(question) > 0:
            llm_start(question)