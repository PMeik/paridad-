"""
core.py — el corazón compartido del proyecto Paridad.

Una sola función, juzgar(), que el motor y la sonda reusan:
recibe un texto (artículos de una ley, o párrafos de una política de privacidad)
+ un requisito de la rúbrica, y devuelve un veredicto estructurado en JSON.

Antes de armar nada, corré este archivo directo (python core.py) para
confirmar que tu API key conecta. Si el test pasa, el resto del pipeline
se cuelga de acá sin sorpresas de autenticación.

NOTA DE REPRODUCIBILIDAD:
El juez corre con temperature=0.0 (ver TEMPERATURA abajo) para maximizar la
consistencia entre corridas. La API de Anthropic NO expone un parámetro `seed`
y, según su propia documentación, ni siquiera a temperature=0 el resultado es
100% determinista (drift de punto flotante en GPU). Por eso temperature=0 es
el control correcto y único disponible; pequeñas variaciones en celdas límite
siguen siendo posibles entre corridas y así se declara en el paper.
"""

import os
import json
from dotenv import load_dotenv
import anthropic

# Carga automática del .env (no hace falta exportar la key a mano en la terminal).
load_dotenv()

# La key se lee del entorno, NUNCA hardcodeada en el código.
# Editá el archivo .env y pegá tu key ahí. El .env está en .gitignore.
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# Modelo barato para debuggear; subí a uno más fuerte para la corrida final.
MODELO_DEBUG = "claude-haiku-4-5-20251001"
MODELO_FINAL = "claude-sonnet-4-6"

# Umbral bajo el cual una celda se marca para revisión humana (gold set).
UMBRAL_REVISION = 0.6

# Temperatura del juez. 0.0 = máxima reproducibilidad (tarea analítica, no
# creativa). La API de Anthropic no acepta `seed`; temperature es el único
# control de determinismo. Documentado en el README.
TEMPERATURA = 0.0


def _llamar(prompt: str, modelo: str = MODELO_DEBUG, max_tokens: int = 1000) -> str:
    """Llamada cruda a la API. Devuelve el texto de la respuesta."""
    resp = client.messages.create(
        model=modelo,
        max_tokens=max_tokens,
        temperature=TEMPERATURA,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def _parsear_json(texto: str) -> dict:
    """
    Parseo robusto: el modelo a veces igual envuelve el JSON en ```json ... ```
    aunque le pidas que no. Limpiamos los backticks y parseamos.
    Si falla, devolvemos una celda marcada para revisión en vez de crashear.
    """
    limpio = texto.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(limpio)
    except json.JSONDecodeError:
        return {
            "fallo": "error_parseo",
            "articulo": None,
            "evidence_snippet": None,
            "confidence": 0.0,
            "needs_human_review": True,
            "_raw": texto,  # guardamos el crudo para inspeccionar a mano
        }


def juzgar(texto_fuente: str, requisito: str, modelo: str = MODELO_DEBUG) -> dict:
    """
    El juez LLM, compartido entre motor y sonda.

    texto_fuente: los chunks de ley recuperados (motor) o los párrafos de
                  la política de privacidad (sonda).
    requisito:    un ítem de la rúbrica de tu coágulo, en texto.

    Devuelve un dict con: fallo (full/partial/absent), articulo,
    evidence_snippet (≤15 palabras, ancla antialucinación), confidence,
    y needs_human_review.
    """
    prompt = f"""Eres un asistente jurídico que evalúa si un texto legal satisface un requisito concreto.

REQUISITO A EVALUAR:
{requisito}

TEXTO FUENTE (la ley o política a analizar):
{texto_fuente}

Decide si el texto fuente CUMPLE el requisito y responde SOLO con un objeto JSON,
sin markdown, sin backticks, sin texto antes ni después. Esquema exacto:

{{
  "fallo": "full" | "partial" | "absent",
  "articulo": "el número de artículo/sección donde se ancla, o null si absent",
  "evidence_snippet": "extracto textual literal de máximo 15 palabras que justifica el fallo, o null si absent",
  "confidence": un número entre 0.0 y 1.0,
  "razon": "una frase breve explicando el fallo"
}}

Reglas:
- El evidence_snippet DEBE ser texto copiado literal del texto fuente, no inventado.
- Si no podés anclar la evidencia en texto real, el fallo es "absent".
- confidence bajo si el texto es ambiguo o solo cubre el requisito parcialmente."""

    respuesta = _llamar(prompt, modelo=modelo)
    veredicto = _parsear_json(respuesta)

    # Cableado del needs_human_review según confidence (si no vino ya marcado).
    if "needs_human_review" not in veredicto:
        conf = veredicto.get("confidence", 0.0)
        veredicto["needs_human_review"] = conf < UMBRAL_REVISION

    return veredicto


# ---------------------------------------------------------------------------
# TEST DE CONEXIÓN — corré "python core.py" antes de armar el resto.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Probando conexión con la API...\n")

    ley_demo = """Art. 18. El titular de los datos personales tiene derecho a
    obtener del responsable la eliminación de sus datos personales cuando los
    datos ya no sean necesarios para los fines que motivaron su tratamiento."""

    requisito_demo = "¿La ley reconoce el derecho de supresión/borrado de datos personales?"

    resultado = juzgar(ley_demo, requisito_demo)

    print("Conexión OK. Veredicto del juez sobre el ejemplo:\n")
    print(json.dumps(resultado, indent=2, ensure_ascii=False))