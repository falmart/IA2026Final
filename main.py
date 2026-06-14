"""
CLI principal del segmentador de contenido.

Uso:
    python main.py --model /ruta/al/modelo.gguf
    python main.py --model /ruta/al/modelo.gguf --file texto.txt
    python main.py --model /ruta/al/modelo.gguf --max-seg 5
"""
import argparse
import sys
import textwrap

from llm_client import load_model, score_coherence
from segmenter import segment_dp
from text_utils import split_sentences


SEPARADOR = "─" * 60
DEFAULT_MODEL = "/Users/ingram/qwen2.5-3b-instruct-q4_k_m.gguf"


def print_results(segments: list[list[str]]) -> None:
    print(f"\n{'═'*60}")
    print(f"  SEGMENTACIÓN ÓPTIMA  ({len(segments)} segmentos)")
    print(f"{'═'*60}\n")
    for i, seg in enumerate(segments, 1):
        text = " ".join(seg)
        wrapped = textwrap.fill(text, width=70, subsequent_indent="    ")
        print(f"▶ Segmento {i} ({len(seg)} oración{'es' if len(seg)!=1 else ''}):")
        print(f"    {wrapped}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Segmentación óptima de contenido usando un LLM local."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Ruta al archivo .gguf del modelo local.",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Archivo de texto a segmentar (si no se provee, modo interactivo).",
    )
    parser.add_argument(
        "--max-seg",
        type=int,
        default=6,
        help="Máximo de oraciones por segmento (default: 6).",
    )
    parser.add_argument(
        "--min-seg",
        type=int,
        default=1,
        help="Mínimo de oraciones por segmento (default: 1).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Hilos de CPU para el modelo (default: 4).",
    )
    args = parser.parse_args()

    load_model(args.model, n_threads=args.threads)

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    else:
        print("\n=== Segmentador de Contenido con LLM Local ===")
        print("Ingresa el texto a segmentar (termina con una línea que contenga solo 'FIN'):\n")
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.strip().upper() == "FIN":
                break
            lines.append(line)
        text = "\n".join(lines)

    if not text.strip():
        print("Error: no se proporcionó texto.", file=sys.stderr)
        sys.exit(1)

    sentences = split_sentences(text)
    print(f"\nTexto dividido en {len(sentences)} oraciones:")
    for i, s in enumerate(sentences, 1):
        print(f"  {i:>2}. {s}")

    segments = segment_dp(
        sentences,
        score_fn=score_coherence,
        max_segment_size=args.max_seg,
        min_segment_size=args.min_seg,
    )

    print_results(segments)


if __name__ == "__main__":
    main()
