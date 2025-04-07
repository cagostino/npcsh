import os
import numpy as np
import tempfile
import threading
import time
import queue
import re
import json
import torch
import pyaudio
import wave
from typing import Optional, List, Dict, Any
from gtts import gTTS
from faster_whisper import WhisperModel
import pygame

# Audio Configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 512

# State Management
is_speaking = False
should_stop_speaking = False
tts_sequence = 0
recording_data = []
buffer_data = []
is_recording = False
last_speech_time = 0
running = True

# Queues
audio_queue = queue.Queue()
tts_queue = queue.PriorityQueue()
cleanup_files = []

# Initialize pygame mixer
pygame.mixer.quit()
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

# Device selection
device = "cpu"
print(f"Using device: {device}")

# Load VAD model
print("Loading Silero VAD model...")
vad_model, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    onnx=False,
    verbose=False,
)
vad_model.to(device)

# Load Whisper model
print("Loading Whisper model...")
whisper_model = WhisperModel("base", device=device, compute_type="int8")

# Conversation History Management
history = []
max_history = 10
memory_file = "conversation_history.json"

# Import the get_llm_response function
from npcsh.llm_funcs import get_llm_response
from npcsh.npc_sysenv import NPCSH_CHAT_PROVIDER, NPCSH_CHAT_MODEL, NPCSH_API_URL
import subprocess

print(NPCSH_CHAT_PROVIDER, NPCSH_CHAT_MODEL, NPCSH_API_URL)

import subprocess


def convert_mp3_to_wav(mp3_file, wav_file):
    try:
        # Ensure the output file doesn't exist before conversion
        if os.path.exists(wav_file):
            os.remove(wav_file)

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                mp3_file,
                "-acodec",
                "pcm_s16le",
                "-ac",
                "1",
                "-ar",
                "44100",
                wav_file,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error converting MP3 to WAV: {e.stderr}")
        raise
    except Exception as e:
        print(f"Unexpected error during conversion: {e}")
        raise


# Check if FFmpeg is available
def check_ffmpeg():
    try:
        subprocess.run(
            ["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


has_ffmpeg = check_ffmpeg()

# Device selection
device = "cpu"
print(f"Using device: {device}")

# Load VAD model
print("Loading Silero VAD model...")
vad_model, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    onnx=False,
    verbose=False,
)
vad_model.to(device)

# Load Whisper model
print("Loading Whisper model...")
whisper_model = WhisperModel("base", device=device, compute_type="int8")

# Conversation History Management
history = []
max_history = 10
memory_file = "conversation_history.json"


# History Management Functions
def load_history():
    global history
    try:
        if os.path.exists(memory_file):
            with open(memory_file, "r") as f:
                history = json.load(f)
    except Exception as e:
        print(f"Error loading conversation history: {e}")
        history = []


def save_history():
    try:
        with open(memory_file, "w") as f:
            json.dump(history, f)
    except Exception as e:
        print(f"Error saving conversation history: {e}")


def add_exchange(user_input, assistant_response):
    global history
    exchange = {
        "user": user_input,
        "assistant": assistant_response,
        "timestamp": time.time(),
    }
    history.append(exchange)
    if len(history) > max_history:
        history.pop(0)
    save_history()


def get_context_string():
    context = []
    for exchange in history:
        context.append(f"User: {exchange['user']}")
        context.append(f"Assistant: {exchange['assistant']}")
    return "\n".join(context)


# Audio Management Functions
def cleanup_temp_files():
    global cleanup_files
    for file in list(cleanup_files):
        try:
            if os.path.exists(file):
                os.remove(file)
                cleanup_files.remove(file)
        except Exception:
            pass


def interrupt_speech():
    global should_stop_speaking
    should_stop_speaking = True
    pygame.mixer.music.stop()
    pygame.mixer.music.unload()

    while not tts_queue.empty():
        try:
            _, temp_filename = tts_queue.get_nowait()
            try:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
            except:
                if temp_filename not in cleanup_files:
                    cleanup_files.append(temp_filename)
        except queue.Empty:
            break

    global tts_sequence
    tts_sequence = 0


def audio_callback(in_data, frame_count, time_info, status):
    audio_queue.put(in_data)
    return (in_data, pyaudio.paContinue)


def transcribe_recording(audio_data):
    if not audio_data:
        return None

    audio_np = (
        np.frombuffer(b"".join(audio_data), dtype=np.int16).astype(np.float32) / 32768.0
    )
    return run_transcription(audio_np)


def run_transcription(audio_np):
    try:
        temp_file = os.path.join(
            tempfile.gettempdir(), f"temp_recording_{int(time.time())}.wav"
        )
        with wave.open(temp_file, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes((audio_np * 32768).astype(np.int16).tobytes())

        segments, info = whisper_model.transcribe(temp_file, language="en", beam_size=5)
        transcription = " ".join([segment.text for segment in segments])

        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception:
            if temp_file not in cleanup_files:
                cleanup_files.append(temp_file)

        return transcription.strip()

    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return None


# History Management Functions
def load_history():
    global history
    try:
        if os.path.exists(memory_file):
            with open(memory_file, "r") as f:
                history = json.load(f)
    except Exception as e:
        print(f"Error loading conversation history: {e}")
        history = []


def save_history():
    try:
        with open(memory_file, "w") as f:
            json.dump(history, f)
    except Exception as e:
        print(f"Error saving conversation history: {e}")


def add_exchange(user_input, assistant_response):
    global history
    exchange = {
        "user": user_input,
        "assistant": assistant_response,
        "timestamp": time.time(),
    }
    history.append(exchange)
    if len(history) > max_history:
        history.pop(0)
    save_history()


def get_context_string():
    context = []
    for exchange in history:
        context.append(f"User: {exchange['user']}")
        context.append(f"Assistant: {exchange['assistant']}")
    return "\n".join(context)


# Audio Management Functions
def cleanup_temp_files():
    global cleanup_files
    for file in list(cleanup_files):
        try:
            if os.path.exists(file):
                os.remove(file)
                cleanup_files.remove(file)
        except Exception:
            pass


def interrupt_speech():
    global should_stop_speaking, response_generator, is_speaking, tts_sequence
    should_stop_speaking = True
    pygame.mixer.music.stop()
    pygame.mixer.music.unload()

    while not tts_queue.empty():
        try:
            _, temp_filename = tts_queue.get_nowait()
            try:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
            except:
                if temp_filename not in cleanup_files:
                    cleanup_files.append(temp_filename)
        except queue.Empty:
            break

    tts_sequence = 0
    is_speaking = False


def audio_callback(in_data, frame_count, time_info, status):
    audio_queue.put(in_data)
    return (in_data, pyaudio.paContinue)


# Text-to-Speech Functions
def play_audio_from_queue():
    global is_speaking, cleanup_files, should_stop_speaking
    next_sequence = 0

    while True:
        if should_stop_speaking:
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()

            while not tts_queue.empty():
                try:
                    _, temp_filename = tts_queue.get_nowait()
                    try:
                        if os.path.exists(temp_filename):
                            os.remove(temp_filename)
                    except:
                        if temp_filename not in cleanup_files:
                            cleanup_files.append(temp_filename)
                except queue.Empty:
                    break

            next_sequence = 0
            is_speaking = False
            should_stop_speaking = False
            time.sleep(0.1)
            continue

        try:
            if not tts_queue.empty():
                sequence, temp_filename = tts_queue.queue[0]

                if sequence == next_sequence:
                    sequence, temp_filename = tts_queue.get()
                    is_speaking = True

                    try:
                        if len(cleanup_files) > 0 and not pygame.mixer.music.get_busy():
                            cleanup_temp_files()

                        if should_stop_speaking:
                            continue

                        pygame.mixer.music.load(temp_filename)
                        pygame.mixer.music.play()

                        while (
                            pygame.mixer.music.get_busy() and not should_stop_speaking
                        ):
                            pygame.time.wait(50)

                        pygame.mixer.music.unload()

                    except Exception as e:
                        print(f"Audio playback error: {str(e)}")
                    finally:
                        try:
                            if os.path.exists(temp_filename):
                                os.remove(temp_filename)
                        except:
                            if temp_filename not in cleanup_files:
                                cleanup_files.append(temp_filename)

                        if not should_stop_speaking:
                            next_sequence += 1
                        is_speaking = False

            time.sleep(0.05)
        except Exception:
            time.sleep(0.05)


import pygame
from gtts import gTTS
import tempfile
import os
import logging

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

import pyaudio
import wave
from gtts import gTTS
import tempfile
import os
import logging

import tempfile
import uuid


def create_and_queue_audio(text, state):
    """Create and queue audio with state awareness for TTS/recording coordination"""
    # Set TTS speaking flag
    state["tts_is_speaking"] = True

    if not text.strip():
        print("Empty text, skipping TTS")
        state["tts_is_speaking"] = False
        return

    try:
        unique_id = uuid.uuid4()
        with tempfile.TemporaryDirectory() as temp_dir:
            mp3_file = os.path.join(temp_dir, f"temp_{unique_id}.mp3")
            wav_file = os.path.join(temp_dir, f"temp_{unique_id}.wav")

            tts = gTTS(text=text, lang="en", slow=False)
            tts.save(mp3_file)

            convert_mp3_to_wav(mp3_file, wav_file)

            # Play audio and wait for completion
            play_audio(wav_file, state)
    except Exception as e:
        print(f"Error in TTS process: {e}")
    finally:
        # Ensure flag is reset even if there's an error
        state["tts_is_speaking"] = False
        state["tts_just_finished"] = True

        for file in [mp3_file, wav_file]:
            try:
                if os.path.exists(file):
                    os.remove(file)
            except Exception as e:
                print(f"Error removing temporary file {file}: {e}")


def play_audio(filename, state):
    """Play audio with state awareness for TTS/recording coordination"""
    CHUNK = 4096  # Increased chunk size

    wf = wave.open(filename, "rb")
    p = pyaudio.PyAudio()

    stream = p.open(
        format=p.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True,
    )

    data = wf.readframes(CHUNK)

    # This is blocking until audio is done playing
    while data and state["running"]:  # Check if system still running
        stream.write(data)
        data = wf.readframes(CHUNK)

    stream.stop_stream()
    stream.close()
    p.terminate()

    try:
        os.unlink(filename)
    except:
        pass


def select_model():
    models = [
        "gpt-4o-mini",
        "claude-haiku-3-5-latest",
    ]

    while True:
        try:
            choice = input(
                "\nSelect a model number (or press Enter for default): "
            ).strip()
            if not choice:
                return models[0]["name"]

            choice = int(choice)
            if 1 <= choice <= len(models):
                selected_model = models[choice - 1]["name"]
                print(f"Selected model: {selected_model}")
                return selected_model
            else:
                print(f"Please enter a number between 1 and {len(models)}")
        except ValueError:
            print("Please enter a valid number")
        except Exception as e:
            print(f"Error selecting model: {str(e)}")
            if models:
                return models[0]["name"]
            return "gemma:2b"


def process_response_chunk(text_chunk):
    if not text_chunk.strip():
        return
    processed_text = process_text_for_tts(text_chunk)
    create_and_queue_audio(processed_text)


def process_text_for_tts(text):
    text = re.sub(r"[*<>{}()\[\]&%#@^_=+~]", "", text)
    text = text.strip()
    text = re.sub(r"(\w)\.(\w)\.", r"\1 \2 ", text)
    text = re.sub(r"([.!?])(\w)", r"\1 \2", text)
    return text


"""

To use this code, you'll need to have the following dependencies installed:

```bash
pip install numpy torch torchaudio faster-whisper pygame pyaudio gtts ollama
```

And optionally FFmpeg for audio speed adjustment:
```bash
# On Ubuntu/Debian
sudo apt-get install ffmpeg

# On MacOS with Homebrew
brew install ffmpeg

# On Windows with Chocolatey
choco install ffmpeg
```


"""
