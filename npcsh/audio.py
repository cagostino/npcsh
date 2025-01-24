# Move optional imports into try/except
try:
    import whisper
    from playsound import playsound
    from gtts import gTTS
    import pyaudio
except Exception as e:
    print(f"Error importing audio dependencies: {e}")

import numpy as np
import tempfile
import os
import time
from typing import Optional, List
from .llm_funcs import get_llm_response


def get_audio_level(audio_data):
    return np.max(np.abs(np.frombuffer(audio_data, dtype=np.int16)))


def calibrate_silence(sample_rate=16000, duration=2):
    """
    Function Description:
        This function calibrates the silence level for audio recording.
    Args:
        None
    Keyword Args:
        sample_rate: The sample rate for audio recording.
        duration: The duration in seconds for calibration.
    Returns:
        The silence threshold level.
    """

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=1024,
    )

    print("Calibrating silence level. Please remain quiet...")
    levels = []
    for _ in range(int(sample_rate * duration / 1024)):
        data = stream.read(1024)
        levels.append(get_audio_level(data))

    stream.stop_stream()
    stream.close()
    p.terminate()

    avg_level = np.mean(levels)
    silence_threshold = avg_level * 1.5  # Set threshold slightly above average
    print(f"Silence threshold set to: {silence_threshold}")
    return silence_threshold


def is_silent(audio_data: bytes, threshold: float) -> bool:
    """
    Function Description:
        This function checks if audio data is silent based on a threshold.
    Args:
        audio_data: The audio data to check.
        threshold: The silence threshold level.
    Keyword Args:
        None
    Returns:
        A boolean indicating whether the audio is silent.
    """

    return get_audio_level(audio_data) < threshold


def record_audio(
    sample_rate: int = 16000,
    max_duration: int = 10,
    silence_threshold: Optional[float] = None,
) -> bytes:
    """
    Function Description:
        This function records audio from the microphone.
    Args:
        None
    Keyword Args:
        sample_rate: The sample rate for audio recording.
        max_duration: The maximum duration in seconds.
        silence_threshold: The silence threshold level.
    Returns:
        The recorded audio data.
    """

    if silence_threshold is None:
        silence_threshold = calibrate_silence()

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=1024,
    )

    print("Listening... (speak now)")
    frames = []
    silent_chunks = 0
    has_speech = False
    max_silent_chunks = int(sample_rate * 3.0 / 1024)  # 3.0 seconds of silence
    max_chunks = int(sample_rate * max_duration / 1024)  # Maximum duration in chunks

    start_time = time.time()
    for _ in range(max_chunks):
        data = stream.read(1024)
        frames.append(data)

        if is_silent(data, silence_threshold):
            silent_chunks += 1
            if has_speech and silent_chunks > max_silent_chunks:
                break
        else:
            silent_chunks = 0
            has_speech = True

        if len(frames) % 10 == 0:  # Print a dot every ~0.5 seconds
            print(".", end="", flush=True)

        if time.time() - start_time > max_duration:
            print("\nMax duration reached.")
            break

    print("\nProcessing...")

    stream.stop_stream()
    stream.close()
    p.terminate()

    return b"".join(frames)


def speak_text(text: str) -> None:
    """
    Function Description:
        This function converts text to speech and plays the audio.
    Args:
        text: The text to convert to speech.
    Keyword Args:
        None
    Returns:
        None
    """

    try:
        tts = gTTS(text=text, lang="en")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            playsound(fp.name)
        os.unlink(fp.name)
    except Exception as e:
        print(f"Text-to-speech error: {e}")


def process_audio(file_path: str, table_name: str) -> List:
    """
    Function Description:
        This function is used to process an audio file.
    Args:
        file_path : str : The file path.
        table_name : str : The table name.
    Keyword Args:
        None
    Returns:
        List : The embeddings and texts.
    """

    embeddings = []
    texts = []
    try:
        audio, sr = librosa.load(file_path)
        # Transcribe audio using Whisper
        model = whisper.load_model("base")  # Or a larger model if available
        result = model.transcribe(file_path)
        transcribed_text = result["text"].strip()

        # Split transcribed text into chunks (adjust chunk_size as needed)
        chunk_size = 1000
        for i in range(0, len(transcribed_text), chunk_size):
            chunk = transcribed_text[i : i + chunk_size]
            text_embedding_response = get_llm_response(
                f"Generate an embedding for: {chunk}",
                model="text-embedding-ada-002",
                provider="openai",
            )  # Use a text embedding model
            if (
                isinstance(text_embedding_response, dict)
                and "error" in text_embedding_response
            ):
                print(
                    f"Error generating text embedding: {text_embedding_response['error']}"
                )
            else:
                embeddings.append(text_embedding_response)  # Store the embedding
                texts.append(chunk)  # Store the corresponding text chunk

        return embeddings, texts

    except Exception as e:
        print(f"Error processing audio: {e}")
        return [], []  # Return empty lists in case of error
