"""
cargador.py — convierte un PDF de ley en una lista de chunks por artículo.

Cada chunk es un dict: {"articulo": "14", "texto": "El titular...", "ley": "LGPD"}
El número de artículo es clave: es lo que después permite citar la evidencia
y lo que tu coágulo verifica en el gold set.

Chile usa "Artículo 3°.-" y Brasil usa "Art. 3º". El regex de abajo captura
las dos formas. Si una ley tiene un patrón raro, ajustás PATRON_ARTICULO.
"""

import re
import pdfplumber


# Captura un artículo REAL: al inicio de una línea y con mayúscula inicial
# ("Art. 3º" / "Artículo 3°.-"). NO captura menciones internas en medio de
# una frase ("...conforme o art. 7º...", "...el artículo 19 de la Constitución...")
# porque esas van en minúscula y en medio de la línea.
#
# ^            -> inicio de línea (con flag MULTILINE)
# [ \t]*       -> tolera sangría
# (?:Art\.|Artículo) -> "Art." o "Artículo" con A mayúscula
# \s*\d+       -> el número
PATRON_ARTICULO = re.compile(
    r"(?=^[ \t]*(?:Art\.|Artículo)\s*\.?\s*\d+)",
    re.MULTILINE,  # OJO: sin IGNORECASE, así "art." en minúscula NO corta
)

# Para extraer el número una vez que ya cortamos (acá sí permitimos cualquier caso).
PATRON_NUMERO = re.compile(
    r"^[ \t]*(?:Art\.|Artículo)\s*\.?\s*(\d+)",
)


def pdf_a_texto(ruta_pdf: str) -> str:
    """Extrae todo el texto del PDF como un solo string."""
    paginas = []
    with pdfplumber.open(ruta_pdf) as pdf:
        for p in pdf.pages:
            t = p.extract_text()
            if t:
                paginas.append(t)
    return "\n".join(paginas)


def limpiar(texto: str) -> str:
    """
    Limpieza básica: junta saltos de línea sueltos dentro de un párrafo
    pero respeta los saltos dobles. Sacá acá encabezados/pies repetidos
    si los ves (ej. 'Biblioteca del Congreso Nacional', números de página).
    """
    # Ejemplo de limpieza de ruido repetido — descomentá y ajustá si aparece:
    # texto = texto.replace("Biblioteca del Congreso Nacional de Chile", "")
    # texto = re.sub(r"\n\s*\d+\s*\n", "\n", texto)  # números de página sueltos
    return texto


def trocear(texto: str, ley: str) -> list[dict]:
    """
    Corta el texto en artículos. Devuelve lista de chunks con su número.
    """
    pedazos = PATRON_ARTICULO.split(texto)
    chunks = []
    for pedazo in pedazos:
        pedazo = pedazo.strip()
        if not pedazo:
            continue
        m = PATRON_NUMERO.search(pedazo)
        if not m:
            # texto antes del primer artículo (preámbulo) — lo saltamos
            continue
        chunks.append({
            "ley": ley,
            "articulo": m.group(1),
            "texto": pedazo,
        })
    return chunks


def cargar_ley(ruta_pdf: str, nombre_ley: str) -> list[dict]:
    """Pipeline completo: PDF → texto → limpio → chunks por artículo."""
    texto = pdf_a_texto(ruta_pdf)
    texto = limpiar(texto)
    chunks = trocear(texto, nombre_ley)
    return chunks


# ---------------------------------------------------------------------------
# TEST — corré "python cargador.py" para ver cómo quedó el troceo.
# Ajustá las rutas a tus PDFs reales.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Cambiá estas rutas por las de tus PDFs
    pruebas = [
        ("data/leyes/lgpd_brasil.pdf", "LGPD"),
        ("data/leyes/ley_21719_chile.pdf", "Chile21719"),
    ]

    for ruta, nombre in pruebas:
        print(f"\n{'='*60}\n{nombre}  ({ruta})\n{'='*60}")
        try:
            chunks = cargar_ley(ruta, nombre)
            print(f"Artículos detectados: {len(chunks)}\n")
            # Mostramos los primeros 3 para verificar que el corte salió bien
            for c in chunks[-3:]:
                preview = c["texto"][:120].replace("\n", " ")
                print(f"  Art. {c['articulo']}: {preview}...")
        except FileNotFoundError:
            print(f"  No encontré el archivo. Poné el PDF en {ruta}")