import re


def parse_accept_language(accept_language):
    pattern = re.compile(r'([a-zA-Z]+)(?:-[a-zA-Z]+)*(?:;q=(\d+(?:\.\d+)?))?')
    return [(lang, float(quality) if quality else 1.0) for lang, quality in pattern.findall(accept_language)]


def get_preferred_language(accept_language, supported_languages):
    language_preferences = parse_accept_language(accept_language)
    for lang, _ in sorted(language_preferences, key=lambda x: x[1], reverse=True):
        if lang in supported_languages:
            return lang
    return None
