"""
embeddings.py — el retrieval del motor.

Para cada requisito de la rúbrica, encuentra los k chunks más parecidos
de cada ley. Después esos chunks (pocos, relevantes) se le pasan al juez
LLM de core.py para que clasifique la cobertura.

Modelo: paraphrase-multilingual-mpnet-base-v2
- Multilingüe (clave: tenemos español de Chile + portugués de Brasil)
- ~470 MB, corre en CPU sin GPU
- La primera corrida descarga el modelo desde HuggingFace, después queda cacheado
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from cargador import cargar_ley


MODELO_EMB = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
K_VECINOS = 7  # cuántos artículos se le pasan al juez por requisito
               # Subido de 5 a 7 tras análisis de mismatches: en R07 Chile y
               # R11 LGPD el artículo correcto del ground truth quedaba justo
               # fuera del top 5 (ranking 6-7). Con k=7 el juez los ve.


# ---------------------------------------------------------------------------
# Carga el modelo una sola vez (es global del módulo).
# Si lo importás desde otro archivo, esto se ejecuta una vez al importar.
# ---------------------------------------------------------------------------
print(f"Cargando modelo de embeddings ({MODELO_EMB})...")
modelo = SentenceTransformer(MODELO_EMB)
print("Modelo cargado.\n")


def cargar_rubrica(ruta: str = "data/rubrica.json") -> list[dict]:
    """Devuelve la lista de requisitos del JSON."""
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)["requisitos"]


def similitud_coseno(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Similitud de coseno entre un vector a y una matriz de vectores b.
    Devuelve un array con un score por cada fila de b.

    En simple: 1.0 = idénticos en significado, 0.0 = no se parecen,
    -1.0 = opuestos. Para textos cortos en el mismo dominio,
    valores >0.4 ya son bastante relevantes.
    """
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return b_norm @ a_norm


def recuperar_k(requisito_emb: np.ndarray,
                chunks: list[dict],
                chunks_emb: np.ndarray,
                k: int = K_VECINOS) -> list[dict]:
    """
    Devuelve los k chunks más parecidos al requisito.
    Cada chunk devuelto incluye su score de similitud.
    """
    scores = similitud_coseno(requisito_emb, chunks_emb)
    # Índices de los k más altos, ordenados de mayor a menor
    top_idx = np.argsort(scores)[::-1][:k]
    resultados = []
    for i in top_idx:
        chunk_con_score = dict(chunks[i])  # copia
        chunk_con_score["score"] = float(scores[i])
        resultados.append(chunk_con_score)
    return resultados


def preparar_ley(ruta_pdf: str, nombre_ley: str) -> tuple[list[dict], np.ndarray]:
    """
    Pipeline: PDF → chunks por artículo → matriz de embeddings.
    Devuelve los chunks y sus embeddings alineados por índice.
    """
    print(f"Cargando {nombre_ley}...")
    chunks = cargar_ley(ruta_pdf, nombre_ley)
    print(f"  {len(chunks)} artículos. Embebiendo...")
    textos = [c["texto"] for c in chunks]
    embeddings = modelo.encode(textos, show_progress_bar=True, convert_to_numpy=True)
    print(f"  Listo. Matriz: {embeddings.shape}\n")
    return chunks, embeddings


# ---------------------------------------------------------------------------
# TEST — corré "python embeddings.py" para ver el retrieval funcionando.
# Te muestra, por cada requisito, los k artículos más parecidos de cada ley
# CON su número de artículo y un preview del texto. Es tu chequeo manual:
# ¿el embedding está trayendo los artículos correctos?
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Cargar y embeber las dos leyes
    leyes = {
        "LGPD":       preparar_ley("data/leyes/lgpd_brasil.pdf",      "LGPD"),
        "Chile21719": preparar_ley("data/leyes/ley_21719_chile.pdf",  "Chile21719"),
    }

    # 2. Cargar y embeber los requisitos de la rúbrica
    requisitos = cargar_rubrica()
    print(f"Embebiendo {len(requisitos)} requisitos de la rúbrica...")
    textos_reqs = [r["requisito"] for r in requisitos]
    reqs_emb = modelo.encode(textos_reqs, show_progress_bar=True, convert_to_numpy=True)
    print()

    # 3. Para los primeros 3 requisitos, mostrar qué artículos recuperó de cada ley.
    #    Cambiá el slice [:3] por [:14] para ver todos.
    print("=" * 70)
    print("PREVIEW DEL RETRIEVAL")
    print("=" * 70)
    for i, req in enumerate(requisitos[:3]):
        print(f"\n[{req['id']}] {req['requisito'][:90]}...")
        print(f"        Estándar: {req['articulo_estandar']}")
        for nombre_ley, (chunks, chunks_emb) in leyes.items():
            print(f"\n  → {nombre_ley}:")
            top = recuperar_k(reqs_emb[i], chunks, chunks_emb, k=3)
            gt = req["ground_truth"].get(nombre_ley, {})
            esperado = gt.get("articulo", "(absent)")
            print(f"    Ground truth: fallo={gt.get('fallo')}, art esperado={esperado}")
            for c in top:
                preview = c["texto"][:80].replace("\n", " ")
                print(f"    [{c['score']:.3f}] Art. {c['articulo']}: {preview}...")