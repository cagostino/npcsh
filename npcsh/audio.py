import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import numpy as np
import tempfile
import threading
import time
import queue
import re
import subprocess
import json
import torch
import pyaudio
import wave
from typing import Optional, List
from gtts import gTTS
from faster_whisper import WhisperModel
import pygame
import ollama

# Audio Configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 512

# State Management
is_speaking = False
should_stop_speaking = False
tts_sequence = 0
response_generator = None
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

# Check if FFmpeg is available
def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    onnx=False,
    verbose=False
)
vad_model.to(device)

# Load Whisper model
print("Loading Whisper model...")
whisper_model = WhisperModel("base", device=device, compute_type="int8")

# Conversation History Management
history = []
max_history = 10
memory_file = "conversation_history.json"
```

Part 2/3 - Core Functions:

```python
# History Management Functions
def load_history():
    global history
    try:
        if os.path.exists(memory_file):
            with open(memory_file, 'r') as f:
                history = json.load(f)
    except Exception as e:
        print(f"Error loading conversation history: {e}")
        history = []

def save_history():
    try:
        with open(memory_file, 'w') as f:
            json.dump(history, f)
    except Exception as e:
        print(f"Error saving conversation history: {e}")

def add_exchange(user_input, assistant_response):
    global history
    exchange = {
        'user': user_input,
        'assistant': assistant_response,
        'timestamp': time.time()
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

def process_audio(vad_threshold=0.4, silence_duration=1.2, buffer_duration=0.65):
    global running, is_recording, recording_data, buffer_data, last_speech_time

    max_buffer_size = int(buffer_duration * RATE / CHUNK)

    while running:
        try:
            data = audio_queue.get(timeout=0.5)
            if data:
                audio_array = np.frombuffer(data, dtype=np.int16)
                audio_float = audio_array.astype(np.float32) / 32768.0

                tensor = torch.from_numpy(audio_float).to(device)
                speech_prob = vad_model(tensor, RATE).item()
                current_time = time.time()

                if speech_prob > vad_threshold:
                    last_speech_time = current_time
                    if not is_recording:
                        is_recording = True
                        print("\nSpeech detected, listening...")
                        recording_data.extend(buffer_data)
                        buffer_data = []
                    recording_data.append(data)
                else:
                    if is_recording:
                        if current_time - last_speech_time > silence_duration:
                            is_recording = False
                            print("Speech ended, transcribing...")
                            transcribe_recording()
                        else:
                            recording_data.append(data)
                    else:
                        buffer_data.append(data)
                        if len(buffer_data) > max_buffer_size:
                            buffer_data.pop(0)
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Audio processing error: {str(e)}")

def transcribe_recording():
    global recording_data

    if not recording_data:
        return

    audio_data = b''.join(recording_data)
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

    recording_data = []

    threading.Thread(
        target=run_transcription,
        args=(audio_np,),
        daemon=True
    ).start()

def run_transcription(audio_np):
    try:
        temp_file = os.path.join(tempfile.gettempdir(), f"temp_recording_{int(time.time())}.wav")
        with wave.open(temp_file, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes((audio_np * 32768).astype(np.int16).tobytes())

        segments, info = whisper_model.transcribe(temp_file, language="en", beam_size=5)
        transcription = " ".join([segment.text for segment in segments])

        if transcription.strip():
            handle_transcription(transcription)

        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception:
            if temp_file not in cleanup_files:
                cleanup_files.append(temp_file)

    except Exception as e:
        print(f"Transcription error: {str(e)}")
# History Management Functions
def load_history():
    global history
    try:
        if os.path.exists(memory_file):
            with open(memory_file, 'r') as f:
                history = json.load(f)
    except Exception as e:
        print(f"Error loading conversation history: {e}")
        history = []

def save_history():
    try:
        with open(memory_file, 'w') as f:
            json.dump(history, f)
    except Exception as e:
        print(f"Error saving conversation history: {e}")

def add_exchange(user_input, assistant_response):
    global history
    exchange = {
        'user': user_input,
        'assistant': assistant_response,
        'timestamp': time.time()
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

def process_audio(vad_threshold=0.4, silence_duration=1.2, buffer_duration=0.65):
    global running, is_recording, recording_data, buffer_data, last_speech_time

    max_buffer_size = int(buffer_duration * RATE / CHUNK)

    while running:
        try:
            data = audio_queue.get(timeout=0.5)
            if data:
                audio_array = np.frombuffer(data, dtype=np.int16)
                audio_float = audio_array.astype(np.float32) / 32768.0

                tensor = torch.from_numpy(audio_float).to(device)
                speech_prob = vad_model(tensor, RATE).item()
                current_time = time.time()

                if speech_prob > vad_threshold:
                    last_speech_time = current_time
                    if not is_recording:
                        is_recording = True
                        print("\nSpeech detected, listening...")
                        recording_data.extend(buffer_data)
                        buffer_data = []
                    recording_data.append(data)
                else:
                    if is_recording:
                        if current_time - last_speech_time > silence_duration:
                            is_recording = False
                            print("Speech ended, transcribing...")
                            transcribe_recording()
                        else:
                            recording_data.append(data)
                    else:
                        buffer_data.append(data)
                        if len(buffer_data) > max_buffer_size:
                            buffer_data.pop(0)
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Audio processing error: {str(e)}")

def transcribe_recording():
    global recording_data

    if not recording_data:
        return

    audio_data = b''.join(recording_data)
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

    recording_data = []

    threading.Thread(
        target=run_transcription,
        args=(audio_np,),
        daemon=True
    ).start()

def run_transcription(audio_np):
    try:
        temp_file = os.path.join(tempfile.gettempdir(), f"temp_recording_{int(time.time())}.wav")
        with wave.open(temp_file, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes((audio_np * 32768).astype(np.int16).tobytes())

        segments, info = whisper_model.transcribe(temp_file, language="en", beam_size=5)
        transcription = " ".join([segment.text for segment in segments])

        if transcription.strip():
            handle_transcription(transcription)

        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception:
            if temp_file not in cleanup_files:
                cleanup_files.append(temp_file)

    except Exception as e:
        print(f"Transcription error: {str(e)}")


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

                        while pygame.mixer.music.get_busy() and not should_stop_speaking:
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

def create_and_queue_audio(text, lang='en'):
    if not text.strip():
        return

    global tts_sequence, has_ffmpeg
    current_sequence = tts_sequence
    tts_sequence += 1

    def process_audio():
        try:
            temp_dir = tempfile.gettempdir()
            timestamp = int(time.time() * 1000)
            temp_filename = os.path.join(temp_dir, f"tts_{timestamp}.mp3")

            tts = gTTS(text=text, lang=lang, tld='co.uk', slow=False)
            tts.save(temp_filename)

            if has_ffmpeg:
                try:
                    sped_up_filename = temp_filename.replace(".mp3", "_sped.mp3")
                    speed_factor = 1.15

                    subprocess.run([
                        "ffmpeg", "-y", "-i", temp_filename, "-filter:a", f"atempo={speed_factor}",
                        "-b:a", "192k", sped_up_filename
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                    os.remove(temp_filename)
                    temp_filename = sped_up_filename
                except Exception as e:
                    print(f"FFmpeg processing error (using normal speed): {str(e)}")

            tts_queue.put((current_sequence, temp_filename))

        except Exception as e:
            print(f"TTS error: {str(e)}")
            try:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
            except:
                pass

    audio_thread = threading.Thread(target=process_audio, daemon=True)
    audio_thread.start()

# Model Selection Functions
def get_available_models():
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error running ollama list: {result.stderr}")
            return []

        lines = result.stdout.strip().split('\n')
        if len(lines) <= 1:
            return []

        model_lines = lines[1:]
        models = []

        for line in model_lines:
            parts = re.split(r'\s{2,}', line.strip())
            if len(parts) >= 3:
                models.append({
                    'name': parts[0].strip(),
                    'id': parts[1].strip() if len(parts) > 1 else "",
                    'size': parts[2].strip() if len(parts) > 2 else "",
                    'modified': ' '.join(parts[3:]) if len(parts) > 3 else ""
                })

        return models

    except Exception as e:
        print(f"Error getting models: {str(e)}")
        return []

def select_model():
    models = get_available_models()

    if not models:
        print("No models found. Make sure Ollama is running.")
        return "gemma:2b"

    print("\nAvailable models:")
    for i, model in enumerate(models, 1):
        name = model.get('name', 'Unknown')
        size = model.get('size', 'Unknown size')
        modified = model.get('modified', 'Unknown date')
        print(f"{i}. {name} ({size}) - Last modified: {modified}")

    while True:
        try:
            choice = input("\nSelect a model number (or press Enter for default): ").strip()
            if not choice:
                return models[0]['name']

            choice = int(choice)
            if 1 <= choice <= len(models):
                selected_model = models[choice-1]['name']
                print(f"Selected model: {selected_model}")
                return selected_model
            else:
                print(f"Please enter a number between 1 and {len(models)}")
        except ValueError:
            print("Please enter a valid number")
        except Exception as e:
            print(f"Error selecting model: {str(e)}")
            if models:
                return models[0]['name']
            return "gemma:2b"

def get_model_default_system_prompt(model_name):
    try:
        result = subprocess.run(['ollama', 'show', 'system', model_name], capture_output=True, text=True)

        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return None
    except Exception as e:
        print(f"Error getting default system prompt: {str(e)}")
        return None

# Main Function
def main():
    global running, selected_model, system_message

    import atexit
    atexit.register(cleanup_temp_files)

    playback_thread = threading.Thread(target=play_audio_from_queue, daemon=True)
    playback_thread.start()

    print("Welcome to Ollama Voice Assistant!")

    if has_ffmpeg:
        print("FFmpeg found - Audio speed adjustment enabled")
    else:
        print("FFmpeg not found - Audio will play at normal speed")

    create_and_queue_audio("Welcome to Ollama Voice Assistant!")

    load_history()

    selected_model = select_model()
    print(f"\nUsing model: {selected_model}")
    create_and_queue_audio(f"Using model {selected_model}")

    default_system_message = get_model_default_system_prompt(selected_model)
    if default_system_message

def main():
    global running, selected_model, system_message

    import atexit
    atexit.register(cleanup_temp_files)

    playback_thread = threading.Thread(target=play_audio_from_queue, daemon=True)
    playback_thread.start()

    print("Welcome to Ollama Voice Assistant!")

    if has_ffmpeg:
        print("FFmpeg found - Audio speed adjustment enabled")
    else:
        print("FFmpeg not found - Audio will play at normal speed")

    create_and_queue_audio("Welcome to Ollama Voice Assistant!")

    load_history()

    selected_model = select_model()
    print(f"\nUsing model: {selected_model}")
    create_and_queue_audio(f"Using model {selected_model}")

    default_system_message = get_model_default_system_prompt(selected_model)
    if default_system_message:
        print(f"\nModel's default system prompt available.")

    system_message = input("Enter system message (or press Enter to use model's default): ").strip()
    system_message = system_message or default_system_message or "You are a helpful assistant."
    system_message += "\nHere's our conversation history for context:\n"

    pyaudio_instance = pyaudio.PyAudio()
    audio_stream = pyaudio_instance.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        stream_callback=audio_callback
    )

    process_thread = threading.Thread(target=process_audio, daemon=True)
    process_thread.start()

    print("\nVoice detection active. Speak, or type a message and press Enter. Type 'quit' to exit.")

    try:
        while True:
            typed_input = input("You: ")
            if typed_input.lower() in ('exit', 'quit', 'goodbye'):
                print("\nGoodbye!")
                create_and_queue_audio("Goodbye!")
                break
            elif typed_input:
                if not handle_transcription(typed_input):
                    break

    except KeyboardInterrupt:
        interrupt_speech()
        print("\nGoodbye!")
        create_and_queue_audio("Goodbye!")
    finally:
        running = False
        if 'audio_stream' in locals():
            audio_stream.stop_stream()
            audio_stream.close()
        if 'pyaudio_instance' in locals():
            pyaudio_instance.terminate()

        time.sleep(2)
        cleanup_temp_files()

def handle_transcription(text):
    global should_stop_speaking, response_generator

    interrupt_speech()

    if text.lower() in ('exit', 'quit', 'goodbye'):
        print("\nDetected exit command.")
        return False

    print(f"\nYou: {text}")

    context = get_context_string()
    context_prompt = f"{system_message}\n{context}\n\nUser: {text}\nAssistant:"

    print("\nAssistant: ", end="", flush=True)

    full_response = ""
    current_text = ""
    buffer = []

    response_generator = ollama.generate(
        model=selected_model,
        prompt=text,
        system=context_prompt,
        stream=True
    )

    try:
        for chunk in response_generator:
            if should_stop_speaking:
                break

            new_text = chunk['response']
            print(new_text, end="", flush=True)

            current_text += new_text
            full_response += new_text

            if re.search(r'[.!?]\s*$', current_text) and len(current_text) >= 30:
                buffer.append(current_text)
                if len(buffer) >= 1:
                    combined_text = " ".join(buffer)
                    process_response_chunk(combined_text)
                    buffer = []
                current_text = ""

        if not should_stop_speaking and (buffer or current_text.strip()):
            remaining_text = " ".join(buffer + [current_text.strip()])
            if remaining_text.strip():
                process_response_chunk(remaining_text)

    except Exception as e:
        print(f"\nError during response generation: {e}")
    finally:
        response_generator = None
        should_stop_speaking = False

    if full_response.strip():
        add_exchange(text, full_response)

    print("")
    return True

def process_response_chunk(text_chunk):
    if not text_chunk.strip():
        return
    processed_text = process_text_for_tts(text_chunk)
    create_and_queue_audio(processed_text)

def process_text_for_tts(text):
    text = re.sub(r'[*<>{}()\[\]&%#@^_=+~]', '', text)
    text = text.strip()
    text = re.sub(r'(\w)\.(\w)\.', r'\1 \2 ', text)
    text = re.sub(r'([.!?])(\w)', r'\1 \2', text)
    return text

if __name__ == "__main__":
    main()


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