"""
Utilidades para dividir texto libre en oraciones.
Usa nltk si está disponible; si no, fallback con regex.
"""
import re


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    try:
        import nltk
        try:
            sentences = nltk.sent_tokenize(text, language="spanish")
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            sentences = nltk.sent_tokenize(text, language="spanish")
        return [s.strip() for s in sentences if s.strip()]
    except ImportError:
        pass

    # Fallback: split por puntuación final
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]
