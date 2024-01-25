#!/usr/bin/env python3

# general
from typing import Any, Dict, List, Optional
import threading
import queue

# speech recognition
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_microphone_live

# text to speech
from TTS.api import TTS
from pydub import AudioSegment
from pydub.playback import play

# large language model
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser


# Configuration
llm_model_file = "openhermes-2.5-mistral-7b.Q5_K_M.gguf"

guide = """
    You are a smart chatbot named Samantha (or Sam for short).
    You are an expert in Data Engineering and Analytics.
    You are friendly, and you like to help people.
"""

template = """
    <|im_start|>system
    {guide}<|im_end|>
    <|im_start|>user
    {question}<|im_end|>
    <|im_start|>assistant
"""

# Model initiations
llm: Optional[LlamaCpp] = None
callback_manager: Any = None
asr_model_id: str = ""
tts_model_id: str = ""
transcriber: Any = None
tts: Any = None
audio_queue = queue.Queue()
mic_active = threading.Event()

def llm_init():
    """ 
    Initialize and load the large language model.
    
    This function initializes and loads the large language model. It sets up the necessary callback manager
    and creates an instance of the LlamaCpp class. The model path, temperature, number of GPU layers, batch size,
    callback manager, and verbosity can be customized as per the requirements.
    """
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
    """ 
    Initialize the automatic speech recognition model.

    This function initializes the automatic speech recognition model by setting the global variables
    `asr_model_id` and `transcriber`. The `asr_model_id` is set to "openai/whisper-tiny.en" and the
    `transcriber` is created using the `pipeline` function from the Hugging Face library.

    Returns: None
    """
    global asr_model_id, transcriber
    asr_model_id = "openai/whisper-tiny.en"
    transcriber = pipeline("automatic-speech-recognition",
                           model=asr_model_id,
                           device="cpu")

def tts_init():
    """ 
    Initialize the automatic text to speech model.

    This function initializes the automatic text to speech model by setting the global variables
    `tts_model_id` and `tts`. The `tts_model_id` is set to "tts_models/en/jenny/jenny" and the
    `tts` variable is initialized with the TTS model using the specified `tts_model_id`.

    """
    global tts_model_id, tts
    tts_model_id = "tts_models/en/jenny/jenny"
    tts = TTS(tts_model_id).to("cpu")

# Speech Recognition
def disable_mic():
    """Disable the microphone."""
    mic_active.clear()

def enable_mic():
    """Enable the microphone."""
    mic_active.set()

def transcribe_mic(chunk_length_s: float) -> str:
    """ 
    Transcribe the audio from a microphone.

    Args: chunk_length_s (float): The length of each audio chunk in seconds.
    Returns: str: The transcribed text from the microphone audio.
    """
    global transcriber
    while not mic_active.is_set():
        pass

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

# Text to Speech
def play_audio():
    """
    Playback thread that plays audio segments from the queue.
    Args: None
    Returns: None
    """
    while True:
        audio_segment = audio_queue.get()
        disable_mic()
        play(audio_segment)
        audio_queue.task_done()
        enable_mic()

def text_to_speech(text: str):
    """
    Converts the given text to speech and plays the audio.
    Args: text (str): The text to be converted to speech.
    Returns: None
    """
    global tts
    audio = tts.tts_to_file(text=text, 
                            return_audio=True, 
                            split_sentences=False)
    
    sentence = AudioSegment.from_wav(audio)
    audio_queue.put(sentence) 

# Large Language Model
def llm_start(question: str):
    """
    Ask LLM a question.
    Args: question (str): The question to ask LLM.
    Returns: None
    """
    global llm, template

    if not question.strip():  # Checks if the question is not just whitespace
        print("\nNo valid question received. LLM will not be invoked.\n")
        return

    prompt = PromptTemplate(template=template, input_variables=["guide", "question"])
    chain = prompt | llm | StrOutputParser()
    chain.invoke({"guide": guide, "question": question}, config={})

class StreamingCustomCallbackHandler(StreamingStdOutCallbackHandler):
    """ Callback handler for LLM streaming """

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """
        Run on new LLM token. Concatenate tokens and print when a sentence is complete.
        Args: token (str): The new token to be processed.
        Returns: None
        """
        if not hasattr(self, 'concatenated_tokens'):
            self.concatenated_tokens = ''
        self.concatenated_tokens += token
        if '.' in token:
            text_to_speech(self.concatenated_tokens)
            self.concatenated_tokens = ''

def main():
    asr_init()
    llm_init()
    tts_init()
    enable_mic()

    playback_thread = threading.Thread(target=play_audio, daemon=True)
    playback_thread.start()

    welcome = "Hi! I'm Sam. Feel free to ask me a question."
    print(welcome)
    while True:
        question = transcribe_mic(chunk_length_s=5.0)
        if len(question) > 0:
            print(f"\n{question}\n")
            llm_start(question)
            print(f"\nCan I help with anything else?\n")

if __name__ == "__main__":
    main()