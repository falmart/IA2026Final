from __future__ import annotations
import argparse
import json
import os
from typing import List

from llm_client import load_model, score_coherence
from segmenter import segment_dp
from text_utils import split_sentences
from dataset import get_all, get_by_id, boundaries_from_segments

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "qwen2.5-3b-instruct-q4_k_m.gguf")


# ── Métricas ────────────────────────────────────────────────────────────────

def _ref_vector(boundaries: List[int], n: int) -> List[int]:
    """Vector binario: 1 en posición de frontera, 0 resto."""
    v = [0] * n
    for b in boundaries:
        v[b] = 1
    return v


def pk(ref_bounds: List[int], hyp_bounds: List[int], n: int, k: int | None = None) -> float:
    if k is None:
        k = max(1, n // (len(ref_bounds) + 1) // 2)
    ref = _ref_vector(ref_bounds, n)
    hyp = _ref_vector(hyp_bounds, n)
    errors = 0
    total = 0
    for i in range(n - k):
        ref_diff = ref[i] != ref[i + k]
        hyp_diff = hyp[i] != hyp[i + k]
        if ref_diff != hyp_diff:
            errors += 1
        total += 1
    return errors / total if total > 0 else 0.0


def windowdiff(ref_bounds: List[int], hyp_bounds: List[int], n: int, k: int | None = None) -> float:
    if k is None:
        k = max(1, n // (len(ref_bounds) + 1) // 2)
    ref = _ref_vector(ref_bounds, n)
    hyp = _ref_vector(hyp_bounds, n)
    errors = 0
    total = 0
    for i in range(n - k):
        ref_count = sum(ref[i:i + k])
        hyp_count = sum(hyp[i:i + k])
        errors += abs(ref_count - hyp_count)
        total += 1
    return errors / total if total > 0 else 0.0


def boundary_f1(ref_bounds: List[int], hyp_bounds: List[int]) -> tuple:
    ref_set = set(ref_bounds)
    hyp_set = set(hyp_bounds)
    tp = len(ref_set & hyp_set)
    precision = tp / len(hyp_set) if hyp_set else 0.0
    recall = tp / len(ref_set) if ref_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ── Evaluación ───────────────────────────────────────────────────────────────

def evaluate_doc(doc: dict, max_seg: int, min_seg: int) -> dict:
    sentences = split_sentences(doc["texto"])
    n = len(sentences)

    pred_segments = segment_dp(
        sentences,
        score_fn=score_coherence,
        max_segment_size=max_seg,
        min_segment_size=min_seg,
    )

    ref_bounds = boundaries_from_segments(doc["segmentos"])
    hyp_bounds = boundaries_from_segments(pred_segments)

    pk_score = pk(ref_bounds, hyp_bounds, n)
    wd_score = windowdiff(ref_bounds, hyp_bounds, n)
    prec, rec, f1 = boundary_f1(ref_bounds, hyp_bounds)

    pred_texts = [" ".join(s) for s in pred_segments]
    ref_texts  = [" ".join(s) for s in doc["segmentos"]]

    return {
        "id": doc["id"],
        "temas": doc["temas"],
        "n_oraciones": n,
        "seg_referencia": len(doc["segmentos"]),
        "seg_predichos": len(pred_segments),
        "Pk": round(pk_score, 4),
        "WindowDiff": round(wd_score, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "fronteras_ref": ref_bounds,
        "fronteras_pred": hyp_bounds,
        "segmentos_pred": pred_texts,
        "segmentos_ref": ref_texts,
    }


# Configuraciones a comparar en el experimento
CONFIGURATIONS = [
    {"nombre": "Baseline uniforme", "max_seg": 4, "min_seg": 4, "baseline": True},
    {"nombre": "DP max_seg=3",      "max_seg": 3, "min_seg": 1, "baseline": False},
    {"nombre": "DP max_seg=6",      "max_seg": 6, "min_seg": 1, "baseline": False},
    {"nombre": "DP max_seg=8",      "max_seg": 8, "min_seg": 1, "baseline": False},
    {"nombre": "DP min_seg=2",      "max_seg": 6, "min_seg": 2, "baseline": False},
]


def baseline_uniform(sentences: List[str], k: int) -> List[List[str]]:
    """Segmentación uniforme: divide en k bloques iguales (sin LLM)."""
    n = len(sentences)
    size = max(1, n // k)
    segs = []
    for i in range(0, n, size):
        segs.append(sentences[i:i + size])
    # fusionar el último bloque si quedó muy pequeño
    if len(segs) > 1 and len(segs[-1]) < size // 2:
        segs[-2].extend(segs.pop())
    return segs


def evaluate_config(docs: list, cfg: dict) -> dict:
    """Evalúa todos los documentos bajo una configuración y devuelve promedios."""
    rows = []
    for doc in docs:
        sentences = split_sentences(doc["texto"])
        n = len(sentences)
        ref_k = len(doc["segmentos"])

        if cfg["baseline"]:
            pred_segments = baseline_uniform(sentences, ref_k)
        else:
            pred_segments = segment_dp(
                sentences,
                score_fn=score_coherence,
                max_segment_size=cfg["max_seg"],
                min_segment_size=cfg["min_seg"],
            )

        ref_bounds = boundaries_from_segments(doc["segmentos"])
        hyp_bounds = boundaries_from_segments(pred_segments)

        pk_score = pk(ref_bounds, hyp_bounds, n)
        wd_score = windowdiff(ref_bounds, hyp_bounds, n)
        _, _, f1 = boundary_f1(ref_bounds, hyp_bounds)
        rows.append({"Pk": pk_score, "WD": wd_score, "F1": f1,
                     "pred": len(pred_segments), "ref": ref_k})

    avg = lambda key: round(sum(r[key] for r in rows) / len(rows), 4)
    return {
        "config": cfg["nombre"],
        "max_seg": cfg.get("max_seg", "-"),
        "min_seg": cfg.get("min_seg", "-"),
        "Pk":  avg("Pk"),
        "WD":  avg("WD"),
        "F1":  avg("F1"),
        "detalles": rows,
    }


def print_comparison_table(comparison: list) -> None:
    header = f"{'Configuración':<25} {'max_seg':>7} {'min_seg':>7} {'Pk':>8} {'WD':>8} {'F1':>8}"
    print(f"\n{'='*65}")
    print("COMPARACIÓN DE CONFIGURACIONES")
    print("="*65)
    print(header)
    print("-"*65)
    for r in comparison:
        print(
            f"{r['config']:<25} {str(r['max_seg']):>7} {str(r['min_seg']):>7} "
            f"{r['Pk']:>8.4f} {r['WD']:>8.4f} {r['F1']:>8.4f}"
        )
    print("="*65)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default=None)
    parser.add_argument("--max-seg", type=int, default=None,
                        help="Si se indica, corre solo esa configuración (sin comparación)")
    parser.add_argument("--min-seg", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", default="resultados.json")
    parser.add_argument("--compare", action="store_true",
                        help="Ejecutar comparación entre todas las configuraciones")
    args = parser.parse_args()

    load_model(MODEL_PATH, n_threads=args.threads)

    docs = [get_by_id(args.id)] if args.id else get_all()

    # ── Modo comparación ────────────────────────────────────────
    if args.compare:
        comparison = []
        for cfg in CONFIGURATIONS:
            print(f"\n→ Evaluando configuración: {cfg['nombre']}")
            result = evaluate_config(docs, cfg)
            comparison.append(result)

        print_comparison_table(comparison)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        print(f"\nResultados guardados en {args.output}")
        return

    # ── Modo evaluación simple ──────────────────────────────────
    max_seg = args.max_seg if args.max_seg else 6
    results = []
    for doc in docs:
        print(f"\n{'='*60}")
        print(f"Evaluando {doc['id']} — temas: {', '.join(doc['temas'])}")
        print("="*60)
        r = evaluate_doc(doc, max_seg, args.min_seg)
        results.append(r)
        print(f"  Pk={r['Pk']}  WD={r['WindowDiff']}  F1={r['F1']}")
        print(f"  Segmentos: predichos={r['seg_predichos']}  referencia={r['seg_referencia']}")

    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"PROMEDIO — "
              f"Pk={sum(r['Pk'] for r in results)/len(results):.4f}  "
              f"WD={sum(r['WindowDiff'] for r in results)/len(results):.4f}  "
              f"F1={sum(r['F1'] for r in results)/len(results):.4f}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResultados guardados en {args.output}")


if __name__ == "__main__":
    main()
