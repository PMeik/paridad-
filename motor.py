"""
motor.py — el pipeline base completo.

Por cada (ley × requisito):
  1. Recupera los k chunks más parecidos al requisito (embeddings)
  2. Se los pasa al juez LLM (core) para que clasifique cobertura
  3. Compara el fallo del motor contra el ground truth de la rúbrica
  4. Guarda todo en scoring.json

Al final imprime la precisión del motor: % de celdas donde el motor
acertó el mismo fallo que la abogada marcó como ground truth.
Esa es la métrica clave de la Dimensión 2 del paper.
"""

import json
import time
from pathlib import Path

from cargador import cargar_ley
from embeddings import modelo, recuperar_k, cargar_rubrica, K_VECINOS
from core import juzgar, MODELO_DEBUG, MODELO_FINAL


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
LEYES = {
    "LGPD":       "data/leyes/lgpd_brasil.pdf",
    "Chile21719": "data/leyes/ley_21719_chile.pdf",
}

# Por defecto debug (Haiku) — más barato mientras afinamos.
# Cambiá a MODELO_FINAL para la corrida final.
MODELO_JUEZ = MODELO_DEBUG

# Cuántos chunks le pasa el retrieval al juez por celda
K = K_VECINOS  # 5 por defecto

# Archivo de salida
SALIDA = "scoring.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def preparar_leyes() -> dict:
    """Carga y embebe todas las leyes una sola vez."""
    leyes_data = {}
    for nombre, ruta in LEYES.items():
        print(f"[{nombre}] cargando y embebiendo...")
        chunks = cargar_ley(ruta, nombre)
        textos = [c["texto"] for c in chunks]
        emb = modelo.encode(textos, show_progress_bar=False, convert_to_numpy=True)
        leyes_data[nombre] = (chunks, emb)
        print(f"  {len(chunks)} artículos embebidos.")
    print()
    return leyes_data


def contexto_para_juez(chunks_recuperados: list[dict]) -> str:
    """
    Arma el texto que se le pasa al juez: los k artículos relevantes
    concatenados con un separador claro, encabezados por su número.
    """
    bloques = []
    for c in chunks_recuperados:
        bloques.append(f"--- Art. {c['articulo']} ---\n{c['texto']}")
    return "\n\n".join(bloques)


def comparar_con_ground_truth(fallo_motor: str, fallo_gt: str) -> str:
    """
    Devuelve 'match' si coinciden, 'mismatch' si no.
    Los fallos posibles son: full, partial, absent, absent_sin_enforcement, error_parseo.
    Tratamos absent y absent_sin_enforcement como equivalentes para la métrica.
    """
    def normalizar(f):
        if f and f.startswith("absent"):
            return "absent"
        return f
    return "match" if normalizar(fallo_motor) == normalizar(fallo_gt) else "mismatch"


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def correr_motor():
    # 1. Cargar leyes (con embeddings)
    leyes = preparar_leyes()

    # 2. Cargar rúbrica
    requisitos = cargar_rubrica()
    print(f"Cargados {len(requisitos)} requisitos.\n")

    # 3. Embebermos los requisitos una sola vez
    print("Embebiendo requisitos...")
    textos_reqs = [r["requisito"] for r in requisitos]
    reqs_emb = modelo.encode(textos_reqs, show_progress_bar=False, convert_to_numpy=True)
    print()

    # 4. Loop sobre (requisito × ley)
    resultados = []
    total = len(requisitos) * len(leyes)
    print(f"Corriendo motor: {total} celdas ({len(requisitos)} reqs × {len(leyes)} leyes)\n")

    i = 0
    for r_idx, req in enumerate(requisitos):
        for ley_nombre, (chunks_ley, emb_ley) in leyes.items():
            i += 1
            print(f"[{i}/{total}] {req['id']} sobre {ley_nombre}...", end=" ", flush=True)

            # Retrieval: top k artículos del requisito en esta ley
            top_k = recuperar_k(reqs_emb[r_idx], chunks_ley, emb_ley, k=K)

            # Contexto para el juez
            contexto = contexto_para_juez(top_k)

            # Llamada al juez
            t0 = time.time()
            try:
                veredicto = juzgar(contexto, req["requisito"], modelo=MODELO_JUEZ)
            except Exception as e:
                print(f"ERROR: {e}")
                veredicto = {
                    "fallo": "error_api",
                    "articulo": None,
                    "evidence_snippet": None,
                    "confidence": 0.0,
                    "needs_human_review": True,
                    "_error": str(e),
                }
            dt = time.time() - t0

            # Ground truth para esta celda
            gt = req["ground_truth"].get(ley_nombre, {})
            fallo_gt = gt.get("fallo")

            # Comparación con ground truth
            match = comparar_con_ground_truth(veredicto.get("fallo"), fallo_gt)

            # Construir el registro
            celda = {
                "requisito_id":     req["id"],
                "requisito":        req["requisito"],
                "dimension":        req["dimension"],
                "capa":             req["capa"],
                "articulo_estandar": req["articulo_estandar"],
                "ley":              ley_nombre,
                # Veredicto del motor
                "motor_fallo":      veredicto.get("fallo"),
                "motor_articulo":   veredicto.get("articulo"),
                "motor_evidencia":  veredicto.get("evidence_snippet"),
                "motor_confidence": veredicto.get("confidence"),
                "motor_razon":      veredicto.get("razon"),
                "needs_human_review": veredicto.get("needs_human_review"),
                # Ground truth (lo que dijo la abogada)
                "gt_fallo":         fallo_gt,
                "gt_articulo":      gt.get("articulo"),
                "gt_razon":         gt.get("razon"),
                # Comparación
                "match_gt":         match,
                # Trazabilidad: qué artículos vio el juez
                "chunks_top_k":     [{"articulo": c["articulo"], "score": c["score"]} for c in top_k],
                "tiempo_seg":       round(dt, 2),
            }
            resultados.append(celda)

            simbolo = "✓" if match == "match" else "✗"
            print(f"{simbolo} motor={veredicto.get('fallo')} gt={fallo_gt} ({dt:.1f}s)")

    # 5. Guardar el JSON completo
    payload = {
        "_meta": {
            "modelo_juez": MODELO_JUEZ,
            "modelo_embeddings": "paraphrase-multilingual-mpnet-base-v2",
            "k_vecinos": K,
            "leyes": list(LEYES.keys()),
            "n_requisitos": len(requisitos),
        },
        "celdas": resultados,
    }
    Path(SALIDA).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 6. Métricas finales
    print()
    print("=" * 70)
    print("RESUMEN")
    print("=" * 70)
    n = len(resultados)
    matches = sum(1 for c in resultados if c["match_gt"] == "match")
    print(f"Celdas totales:    {n}")
    print(f"Matches con GT:    {matches}  ({100*matches/n:.1f}%)")
    print(f"Mismatches:        {n - matches}")
    needs_review = sum(1 for c in resultados if c.get("needs_human_review"))
    print(f"Necesitan revisión humana (confidence baja): {needs_review}")
    print()
    print("Por ley:")
    for ley in LEYES:
        sub = [c for c in resultados if c["ley"] == ley]
        m = sum(1 for c in sub if c["match_gt"] == "match")
        print(f"  {ley}:  {m}/{len(sub)} ({100*m/len(sub):.1f}%)")
    print()
    print(f"Resultados completos en: {SALIDA}")


if __name__ == "__main__":
    correr_motor()
