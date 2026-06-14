"""
Segmentación óptima mediante programación dinámica.

Dado una lista de oraciones, encuentra la partición en segmentos
contiguos que maximice la suma de coherencias (evaluadas por el LLM).

Complejidad: O(n² · tiempo_LLM). Para n > 20 oraciones se recomienda
limitar max_segment_size para reducir llamadas al modelo.
"""
from __future__ import annotations
import math
from typing import Callable

ScoreFn = Callable[[str], float]


def _sentences_to_text(sentences: list[str]) -> str:
    return " ".join(s.strip() for s in sentences)


def segment_dp(
    sentences: list[str],
    score_fn: ScoreFn,
    max_segment_size: int = 6,
    min_segment_size: int = 1,
    length_penalty: float = 0.05,
) -> list[list[str]]:
    """
    Programación dinámica para segmentación óptima.

    Args:
        sentences:         Lista de oraciones del texto.
        score_fn:          Función que recibe texto y devuelve coherencia [0,1].
        max_segment_size:  Máximo de oraciones por segmento (limita llamadas al LLM).
        min_segment_size:  Mínimo de oraciones por segmento.
        length_penalty:    Penalización leve a segmentos de una sola oración.

    Returns:
        Lista de segmentos; cada segmento es una lista de oraciones.
    """
    n = len(sentences)
    if n == 0:
        return []

    # dp[i] = mejor puntuación acumulada para las primeras i oraciones
    dp = [-math.inf] * (n + 1)
    dp[0] = 0.0
    # split[i] = índice donde empieza el segmento que termina en i
    split = [0] * (n + 1)

    total_calls = sum(
        1
        for i in range(n)
        for j in range(i + min_segment_size, min(i + max_segment_size, n) + 1)
    )
    call_count = 0

    print(f"\n→ Evaluando {total_calls} segmentos candidatos con el LLM...\n")

    for i in range(n):
        if dp[i] == -math.inf:
            continue
        for j in range(i + min_segment_size, min(i + max_segment_size, n) + 1):
            seg_sentences = sentences[i:j]
            seg_text = _sentences_to_text(seg_sentences)
            coherence = score_fn(seg_text)

            # Penalización suave a segmentos muy cortos
            size = j - i
            penalty = length_penalty if size == 1 else 0.0
            adjusted = coherence - penalty

            call_count += 1
            print(
                f"  [{call_count}/{total_calls}] ors {i+1}-{j}: "
                f"coherencia={coherence:.3f}  «{seg_text[:60]}{'...' if len(seg_text)>60 else ''}»"
            )

            candidate = dp[i] + adjusted
            if candidate > dp[j]:
                dp[j] = candidate
                split[j] = i

    # Reconstruir segmentos desde split[]
    segments: list[list[str]] = []
    idx = n
    while idx > 0:
        start = split[idx]
        segments.append(sentences[start:idx])
        idx = start
    segments.reverse()
    return segments
