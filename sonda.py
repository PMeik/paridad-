"""
sonda.py — la capa hallazgo del proyecto.

ARQUITECTURA v2 — soporta dos formatos de policies:

1. MULTI-ARCHIVO (estilo OpenAI):
   El proveedor publica documentos separados por jurisdicción.

2. MONOLÍTICA CON ANEXOS REGIONALES (estilo Anthropic):
   Un solo documento con secciones generales + "Divulgación Suplementaria
   Regional" con subsecciones por país.

LOS TRES LOCALES SONDEADOS:
- "eu":     usuario en UE/EEA
- "brasil": usuario en Brasil (puede recibir anexo LGPD si existe)
- "row":    usuario en LatAm-no-Brasil (ej. Chile), recibe solo lo genérico

HALLAZGO DE GRADIENTE:
Si un proveedor tiene anexo Brasil pero no Chile, eso es evidencia de que
la institucionalidad (ANPD operativa) mueve la aguja, mientras que leyes
sin enforcement creíble (21.719 con Agencia en constitución) reciben
trato genérico.
"""

import json
import re
import time
from pathlib import Path

import numpy as np

from core import juzgar, MODELO_DEBUG
from embeddings import modelo


SERVICIOS = {
    "openai": {
        "tipo":   "multi_archivo",
        "eu":     "data/policies/openai/eu_es.txt",
        "brasil": "data/policies/openai/row_es.txt",
        "row":    "data/policies/openai/row_es.txt",
    },
    "anthropic": {
        "tipo":   "monolitico_anexos",
        "ruta":   "data/policies/anthropic/full.txt",
        "marcador_anexos":   r"^11\.\s*Divulgación Suplementaria Regional",
        "marcador_brasil":   r"^Información adicional para residentes en Brasil",
        "marcadores_otros":  [
            r"^Información adicional para residentes en Canadá",
            r"^Información adicional para residentes en la República de Corea",
        ],
    },
}

MODELO_JUEZ   = MODELO_DEBUG
K_PARRAFOS    = 7
RUBRICA_PATH  = "data/rubrica.json"
SALIDA        = "sonda.json"


def trocear_policy(texto: str) -> list[str]:
    bloques = [b.strip() for b in re.split(r"\n\s*\n", texto) if b.strip()]
    return [b for b in bloques if len(b) > 80]


def cargar_y_embeber(texto: str):
    parrafos = trocear_policy(texto)
    if not parrafos:
        return [], np.zeros((0, 768))
    emb = modelo.encode(parrafos, show_progress_bar=False, convert_to_numpy=True)
    return parrafos, emb


def cargar_multi_archivo(config: dict) -> dict:
    out = {}
    for locale in ("eu", "brasil", "row"):
        if locale not in config:
            continue
        texto = Path(config[locale]).read_text(encoding="utf-8")
        out[locale] = cargar_y_embeber(texto)
    return out


def cargar_monolitico_anexos(config: dict) -> dict:
    texto = Path(config["ruta"]).read_text(encoding="utf-8")

    m_anexos = re.search(config["marcador_anexos"], texto, re.MULTILINE)
    if not m_anexos:
        raise ValueError(f"No encontré marcador Sección 11 en {config['ruta']}")
    inicio_anexos = m_anexos.start()

    generales = texto[:inicio_anexos]
    anexos_full = texto[inicio_anexos:]

    m_brasil = re.search(config["marcador_brasil"], anexos_full, re.MULTILINE)
    if not m_brasil:
        anexo_brasil = ""
    else:
        inicio_brasil = m_brasil.start()
        siguiente_idx = len(anexos_full)
        for marcador_otro in config["marcadores_otros"]:
            m = re.search(marcador_otro, anexos_full[inicio_brasil + 50:], re.MULTILINE)
            if m:
                siguiente_idx = min(siguiente_idx, inicio_brasil + 50 + m.start())
        anexo_brasil = anexos_full[inicio_brasil:siguiente_idx]

    out = {}
    out["eu"]     = cargar_y_embeber(generales)
    out["brasil"] = cargar_y_embeber(generales + "\n\n" + anexo_brasil)
    out["row"]    = cargar_y_embeber(generales)
    return out


def cargar_servicio(nombre: str, config: dict) -> dict:
    if config["tipo"] == "multi_archivo":
        return cargar_multi_archivo(config)
    elif config["tipo"] == "monolitico_anexos":
        return cargar_monolitico_anexos(config)
    raise ValueError(f"Tipo desconocido: {config['tipo']}")


def similitud_coseno(a, b):
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return b_norm @ a_norm


def recuperar_parrafos(req_emb, parrafos, parrafos_emb, k=K_PARRAFOS):
    if len(parrafos) == 0:
        return []
    scores = similitud_coseno(req_emb, parrafos_emb)
    top_idx = np.argsort(scores)[::-1][:k]
    return [{"texto": parrafos[i], "score": float(scores[i])} for i in top_idx]


def contexto_para_juez(parrafos_top):
    return "\n\n---\n\n".join(p["texto"] for p in parrafos_top)


CONTEXTO_USUARIO = {
    "eu":     "un usuario residente en la Unión Europea / EEE",
    "brasil": "un usuario residente en Brasil (país con LGPD vigente y ANPD operativa)",
    "row":    "un usuario residente en LatAm FUERA de Brasil, por ejemplo en Chile (cuya Ley 21.719 aún no está vigente y cuya Agencia no está operativa)",
}


def adaptar_requisito(req: str, locale: str) -> str:
    return (
        f"En el contexto de una política de privacidad de un servicio de IA, "
        f"evaluá desde la perspectiva de {CONTEXTO_USUARIO[locale]}: "
        f"¿la policy ofrece o garantiza A ESTE USUARIO el siguiente derecho/protección? "
        f"{req} "
        f"IMPORTANTE: si la policy menciona que un derecho aplica a usuarios de OTRA jurisdicción "
        f"distinta a la de este usuario, NO cuenta como cumplimiento para este usuario."
    )


def correr_sonda():
    with open(RUBRICA_PATH, encoding="utf-8") as f:
        todos = json.load(f)["requisitos"]
    sondeables = [r for r in todos if r.get("es_sonda")]
    print(f"Requisitos sondeables: {len(sondeables)}\n")

    print("Embebiendo requisitos...")
    reqs_emb = modelo.encode(
        [r["requisito"] for r in sondeables],
        show_progress_bar=False, convert_to_numpy=True,
    )
    print()

    policies = {}
    for servicio, config in SERVICIOS.items():
        print(f"[{servicio}] cargando ({config['tipo']})...")
        policies[servicio] = cargar_servicio(servicio, config)
        for locale, (parrafos, _) in policies[servicio].items():
            print(f"  {locale}: {len(parrafos)} párrafos")
    print()

    n_celdas = sum(len(locales) for locales in policies.values()) * len(sondeables)
    print(f"Corriendo sonda: {n_celdas} celdas\n")

    resultados = []
    i = 0
    for servicio, locales_data in policies.items():
        for locale, (parrafos, emb) in locales_data.items():
            for r_idx, req in enumerate(sondeables):
                i += 1
                print(f"[{i}/{n_celdas}] {req['id']} sobre {servicio}/{locale}...",
                      end=" ", flush=True)

                top = recuperar_parrafos(reqs_emb[r_idx], parrafos, emb, k=K_PARRAFOS)
                contexto = contexto_para_juez(top)
                req_adaptado = adaptar_requisito(req["requisito"], locale)

                t0 = time.time()
                try:
                    veredicto = juzgar(contexto, req_adaptado, modelo=MODELO_JUEZ)
                except Exception as e:
                    print(f"ERROR: {e}")
                    veredicto = {
                        "fallo": "error_api", "articulo": None,
                        "evidence_snippet": None, "confidence": 0.0,
                        "needs_human_review": True, "_error": str(e),
                    }
                dt = time.time() - t0

                resultados.append({
                    "requisito_id":      req["id"],
                    "requisito":         req["requisito"],
                    "dimension":         req["dimension"],
                    "capa":              req["capa"],
                    "articulo_estandar": req["articulo_estandar"],
                    "servicio":          servicio,
                    "locale":            locale,
                    "fallo":             veredicto.get("fallo"),
                    "seccion":           veredicto.get("articulo"),
                    "evidencia":         veredicto.get("evidence_snippet"),
                    "confidence":        veredicto.get("confidence"),
                    "razon":             veredicto.get("razon"),
                    "needs_human_review": veredicto.get("needs_human_review"),
                    "parrafos_vistos":   [{"score": p["score"]} for p in top],
                    "tiempo_seg":        round(dt, 2),
                })
                print(f"fallo={veredicto.get('fallo')} conf={veredicto.get('confidence')} ({dt:.1f}s)")

    payload = {
        "_meta": {
            "modelo_juez":       MODELO_JUEZ,
            "modelo_embeddings": "paraphrase-multilingual-mpnet-base-v2",
            "k_parrafos":        K_PARRAFOS,
            "servicios":         list(SERVICIOS.keys()),
            "locales":           ["eu", "brasil", "row"],
            "n_requisitos":      len(sondeables),
            "comparacion":       "es vs es (consistencia lingüística entre locales)",
        },
        "celdas": resultados,
    }
    Path(SALIDA).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("MATRICES DE PARIDAD POR SERVICIO  (eu=europeo, brasil=brasileño, row=chileno)")
    print("=" * 80)
    for servicio in policies.keys():
        print(f"\n--- {servicio.upper()} ---")
        print(f"{'Req':6s} | {'eu':10s} | {'brasil':10s} | {'row(Chile)':12s} | Patrón")
        print("-" * 80)
        for req in sondeables:
            def fallo_de(loc):
                c = next((c for c in resultados
                          if c["requisito_id"] == req["id"]
                          and c["servicio"] == servicio
                          and c["locale"] == loc), None)
                return c["fallo"] if c else "—"
            fe = fallo_de("eu"); fb = fallo_de("brasil"); fr = fallo_de("row")
            if fe == fb == fr:
                patron = "plano"
            elif fe == fb and fb != fr:
                patron = "UE = Brasil > Chile"
            elif fb == fr and fe != fb:
                patron = "UE > Brasil = Chile"
            elif fe != fb and fb != fr and fe != fr:
                patron = "gradiente UE>BR>CL"
            else:
                patron = "mixto"
            print(f"{req['id']:6s} | {fe:10s} | {fb:10s} | {fr:12s} | {patron}")

    print()
    print(f"Resultados completos en: {SALIDA}")


if __name__ == "__main__":
    correr_sonda()