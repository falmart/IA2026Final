import os
import threading
from typing import List, Dict, Any
from flask import Flask, render_template, request, jsonify

from llm_client import load_model, score_coherence
from segmenter import segment_dp
from text_utils import split_sentences

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "qwen2.5-3b-instruct-q4_k_m.gguf")

app = Flask(__name__)

# El modelo se carga una sola vez al iniciar el servidor
_model_ready = False
_model_lock = threading.Lock()


def init_model():
    global _model_ready
    with _model_lock:
        load_model(MODEL_PATH, n_threads=4)
        _model_ready = True


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/segmentar", methods=["POST"])
def segmentar():
    if not _model_ready:
        return jsonify({"error": "El modelo aún se está cargando. Intenta en unos segundos."}), 503

    data = request.get_json()
    texto = (data.get("texto") or "").strip()
    max_seg = int(data.get("max_seg", 6))
    min_seg = int(data.get("min_seg", 1))

    if not texto:
        return jsonify({"error": "El texto no puede estar vacío."}), 400

    sentences = split_sentences(texto)
    if len(sentences) < 2:
        return jsonify({"error": "El texto debe tener al menos 2 oraciones."}), 400

    # Recolectamos puntuaciones intermedias para mostrarlas en la UI
    scores_log: List[Dict[str, Any]] = []

    def scoring_with_log(seg_text: str) -> float:
        score = score_coherence(seg_text)
        scores_log.append({"text": seg_text[:80], "score": round(score, 3)})
        return score

    segments = segment_dp(
        sentences,
        score_fn=scoring_with_log,
        max_segment_size=max_seg,
        min_segment_size=min_seg,
    )

    result_segments = []
    for seg in segments:
        text = " ".join(seg)
        # Obtener la puntuación ya calculada para este segmento exacto
        seg_score = next(
            (entry["score"] for entry in scores_log if entry["text"] == text[:80]),
            None,
        )
        result_segments.append({
            "oraciones": seg,
            "texto": text,
            "num_oraciones": len(seg),
            "score": seg_score,
        })

    return jsonify({
        "total_oraciones": len(sentences),
        "total_segmentos": len(segments),
        "segmentos": result_segments,
        "log": scores_log,
    })


if __name__ == "__main__":
    # Cargar modelo en el hilo principal antes de arrancar Flask
    init_model()
    app.run(debug=False, port=5000, threaded=True)
