from google.cloud import translate_v2 as translate # Using v2 for simplicity

def translate_text(text: str, target_language: str, source_language: str = "en-US") -> str:
    """
    Translates text to the target language using Google Cloud Translation.

    Args:
        text (str): The text to translate.
        target_language (str): The target language code (e.g., "es", "fr").
        source_language (str): The source language code (e.g., "en-US").

    Returns:
        str: The translated text. Returns the original text on error.
    """
    if not text:
        return ""

    translate_client = translate.Client()

    try:
        # The API expects language codes like 'en', 'es', 'fr'.
        # If 'en-US' is passed, it might need to be shortened to 'en'.
        if '-' in source_language:
            source_language = source_language.split('-')[0]
        if '-' in target_language: # Should generally be like 'es', 'fr', not 'es-ES' for target
            target_language = target_language.split('-')[0]

        result = translate_client.translate(
            text,
            target_language=target_language,
            source_language=source_language
        )
        return result['translatedText']
    except Exception as e:
        print(f"Translation Error: {e}")
        return text # Return original text if translation fails
