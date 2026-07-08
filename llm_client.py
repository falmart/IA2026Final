import math
import re
from typing import Optional

from llama_cpp import Llama

_model: Optional[Llama] = None

# Tokens que cuentan como respuesta afirmativa / negativa
_YES_TOKENS = {"sí", "si", "s", "sí,", "si,", "yes"}
_NO_TOKENS = {"no", "n", "no,"}


def load_model(model_path: str, n_ctx: int = 2048, n_threads: int = 4) -> None:
    global _model
    print(f"Cargando modelo desde {model_path} ...")
    _model = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        logits_all=True,  # necesario para leer logprobs del token de respuesta
        verbose=False,
    )
    print("Modelo cargado.")


def score_coherence(segment_text: str) -> float:
    """
    Coherencia semántica del segmento en [0, 1].

    Se pregunta al LLM si todas las oraciones del segmento tratan un mismo
    tema (pregunta binaria Sí/No) y se usa la probabilidad del token "Sí"
    como puntuación continua: P(Sí) / (P(Sí) + P(No)).
    Este esquema es mucho más estable en modelos pequeños que pedir
    directamente un número decimal.
    """
    if _model is None:
        raise RuntimeError("Modelo no cargado. Llama load_model() primero.")

    response = _model.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "Eres un evaluador de cohesión temática. Respondes únicamente Sí o No.",
            },
            {
                "role": "user",
                "content": (
                    "¿Tratan todas las oraciones del siguiente segmento un mismo tema central? "
                    "Responde únicamente Sí o No.\n\n"
                    f"Segmento:\n{segment_text}"
                ),
            },
        ],
        max_tokens=1,
        temperature=0.0,
        logprobs=True,
        top_logprobs=10,
    )

    choice = response["choices"][0]
    logprobs = choice.get("logprobs") or {}
    content = logprobs.get("content") or []

    if content:
        top = content[0].get("top_logprobs", [])
        p_yes = sum(
            math.exp(t["logprob"]) for t in top if t["token"].strip().lower() in _YES_TOKENS
        )
        p_no = sum(
            math.exp(t["logprob"]) for t in top if t["token"].strip().lower() in _NO_TOKENS
        )
        if p_yes + p_no > 0:
            return p_yes / (p_yes + p_no)

    # Fallback: interpretar el texto de la respuesta directamente
    raw = (choice.get("message", {}).get("content") or "").strip().lower()
    if raw in _YES_TOKENS:
        return 1.0
    if raw in _NO_TOKENS:
        return 0.0
    nums = re.findall(r"\d+\.?\d*", raw)
    if nums:
        return max(0.0, min(1.0, float(nums[0])))
    return 0.5  # fallback neutro
