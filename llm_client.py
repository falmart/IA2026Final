from typing import Optional
from llama_cpp import Llama

_model: Optional[Llama] = None


def load_model(model_path: str, n_ctx: int = 2048, n_threads: int = 4) -> None:
    global _model
    print(f"Cargando modelo desde {model_path} ...")
    _model = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        verbose=False,
    )
    print("Modelo cargado.")


def score_coherence(segment_text: str) -> float:
    if _model is None:
        raise RuntimeError("Modelo no cargado. Llama load_model() primero.")

    prompt = (
        "Evalúa la coherencia semántica interna del siguiente segmento de texto. "
        "La coherencia mide si todas las oraciones hablan de un mismo tema o idea central. "
        "Responde ÚNICAMENTE con un número decimal entre 0.0 (sin coherencia) y 1.0 (perfectamente coherente). "
        "No expliques nada, solo el número.\n\n"
        f"Segmento:\n\"\"\"\n{segment_text}\n\"\"\"\n\n"
        "Puntuación:"
    )

    response = _model(
        prompt,
        max_tokens=8,
        temperature=0.0,
        stop=["\n", " ", ","],
    )
    raw = response["choices"][0]["text"].strip()

    try:
        score = float(raw)
        return max(0.0, min(1.0, score))
    except ValueError:
        # Si el modelo no devuelve un número limpio, intentamos extraer el primero
        import re
        nums = re.findall(r"\d+\.?\d*", raw)
        if nums:
            return max(0.0, min(1.0, float(nums[0])))
        return 0.5  # fallback neutro
