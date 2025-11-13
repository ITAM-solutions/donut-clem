from deep_translator import GoogleTranslator


def translate_dict(data: dict, lang: str, keys_and_vals=False):
    return {
        translate(k, lang): v for k, v in data.items()
    }

def translate(text: str, lang: str):
    lang = lang.split('_')[0]

    if lang == 'en':
        return text

    try:
        return GoogleTranslator(source="auto", target=lang).translate(text)
    except Exception:
        return text
