from google.cloud import texttospeech

# Preferred voice mapping (can be expanded)
PREFERRED_TTS_VOICES = {
    "en-US": "en-US-Standard-D", # Example: Male voice
    "en-GB": "en-GB-Standard-A", # Example: Female British voice
    "es-ES": "es-ES-Standard-A",
    "fr-FR": "fr-FR-Standard-C",
    "de-DE": "de-DE-Standard-B",
    # Add more mappings: language_code -> voice_name
}

def synthesize_speech(text: str, language_code: str = "en-US", voice_name: str = None) -> bytes:
    """
    Synthesizes speech from text using Google Cloud Text-to-Speech.

    Args:
        text (str): The text to synthesize.
        language_code (str): The language code (e.g., "en-US", "es-ES").
                                The STT service might output "en-US", translation might output "es".
                                TTS often prefers more specific codes like "es-ES" for regional accents if available.
        voice_name (str, optional): Specific voice name (e.g., "en-US-Wavenet-D").
                                    If None, the service will select a default based on the language code.

    Returns:
        bytes: The synthesized audio content in MP3 format by default. Returns None on error.
    """
    if not text:
        return None

    client = texttospeech.TextToSpeechClient()

    input_text = texttospeech.SynthesisInput(text=text)

    # Construct the voice selection params
    voice_params = texttospeech.VoiceSelectionParams(language_code=language_code)
    if voice_name: # User explicitly provided a voice name
        voice_params.name = voice_name
    elif language_code in PREFERRED_TTS_VOICES: # Check our preferred map
        voice_params.name = PREFERRED_TTS_VOICES[language_code]
    # If not in map and no explicit voice_name, TTS service will use its default for the language.
    # No explicit else needed here, as the service handles defaults.

    # Select the type of audio file you want returned
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3 # MP3 is widely playable
    )

    try:
        response = client.synthesize_speech(
            request={"input": input_text, "voice": voice_params, "audio_config": audio_config}
        )
        return response.audio_content
    except Exception as e:
        print(f"TTS Error for text '{text}' in language '{language_code}': {e}")
        return None
