import openai
import whisper
import pyttsx3

# Initialize Whisper model
model = whisper.load_model(
    "base"
)  # You can choose 'base', 'small', 'medium', or 'large'


# Function for Speech-to-Text using Whisper
def speech_to_text(audio_path):
    result = model.transcribe(audio_path)
    return result["text"]


# Initialize TTS engine (pyttsx3)
engine = pyttsx3.init()


# Function to speak text
def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()


# Example usage
if __name__ == "__main__":
    # Convert speech to text from an audio file
    audio_file = "your_audio_file.wav"  # Path to the audio file
    print("Converting speech to text...")
    text = speech_to_text(audio_file)
    print("Recognized Text:", text)

    # Use GPT-4O Mini to generate a response (for now, just a placeholder example)
    # Ideally, replace this with a GPT-4 call for a response based on 'text'
    gpt_response = "This is the response from GPT-4O Mini."

    # Convert the GPT response to speech
    print("Speaking the response...")
    text_to_speech(gpt_response)
