from google.cloud import speech
import io

def transcribe_audio_chunk(audio_data: bytes, sample_rate: int, language_code: str = "en-US") -> str:
    """
    Transcribes a short audio chunk using Google Cloud Speech-to-Text.

    Args:
        audio_data (bytes): Raw audio data.
        sample_rate (int): The sample rate of the audio data.
        language_code (str): The language code for STT (e.g., "en-US", "es-ES").

    Returns:
        str: The transcribed text. Returns an empty string on error.
    """
    client = speech.SpeechClient()

    audio = speech.RecognitionAudio(content=audio_data)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, # Assuming PCM 16-bit
        sample_rate_hertz=sample_rate,
        language_code=language_code, # Use the parameter
        # model="video", # Optionally use a model optimized for audio from video
    )

    try:
        response = client.recognize(config=config, audio=audio)
        if response.results and response.results[0].alternatives:
            return response.results[0].alternatives[0].transcript
        return ""
    except Exception as e:
        print(f"STT Error: {e}")
        return ""

def transcribe_audio_file_path(audio_file_path: str, sample_rate: int, language_code: str = "en-US") -> str:
    """
    Transcribes an audio file using Google Cloud Speech-to-Text.
    This is more for testing the STT service independently.
    For real-time, transcribe_audio_chunk is preferred.

    Args:
        audio_file_path (str): Path to the audio file.
        sample_rate (int): The sample rate of the audio.
        language_code (str): The language code for STT (e.g., "en-US", "es-ES").

    Returns:
        str: The transcribed text.
    """
    client = speech.SpeechClient()

    with io.open(audio_file_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, # Adjust if not LINEAR16
        sample_rate_hertz=sample_rate,
        language_code=language_code, # Use the parameter
    )

    try:
        response = client.recognize(config=config, audio=audio)
        if response.results and response.results[0].alternatives:
            return response.results[0].alternatives[0].transcript
        return ""
    except Exception as e:
        print(f"STT Error from file: {e}")
        return ""
