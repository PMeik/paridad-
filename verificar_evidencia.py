"""
verificar_evidencia.py — Task 4: evidence-snippet traceability check.

For every cell that carries an evidence_snippet, this script confirms the
snippet can actually be found in its source text (the Ctrl+F test). A snippet
that cannot be located is, by the project's own anti-hallucination rule, a
suspect verdict that must be reviewed by hand.

It checks both pipelines:
  - scoring.json  -> motor_evidencia, anchored in the law PDFs
  - sonda.json    -> evidencia,       anchored in the policy .txt files

Matching is layered so we don't get false alarms from formatting noise:
  1. EXACT        : snippet appears verbatim in the source.
  2. NORMALIZED   : appears after collapsing whitespace, lowercasing and
                    stripping accents (handles PDF line breaks, casing, tildes).
  3. FUZZY        : best contiguous overlap >= --umbral (default 0.85) via
                    difflib; catches snippets the judge lightly trimmed.
  4. NOT FOUND    : none of the above -> flagged SUSPECT for manual review.

Usage:
    python verificar_evidencia.py
    python verificar_evidencia.py --umbral 0.90
    python verificar_evidencia.py --solo-sonda

Requires the source files on disk at the same paths the pipelines use:
    data/leyes/lgpd_brasil.pdf
    data/leyes/ley_21719_chile.pdf
    data/policies/openai/eu_es.txt
    data/policies/openai/row_es.txt
    data/policies/anthropic/full.txt
"""

import argparse
import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path


# --- source locations (must match motor.py / sonda.py) ----------------------
LEYES = {
    "LGPD":       "data/leyes/lgpd_brasil.pdf",
    "Chile21719": "data/leyes/ley_21719_chile.pdf",
}

# For the probe, the source a user actually sees depends on (service, locale).
# Anthropic is monolithic; OpenAI splits eu vs row (brasil shares the row file).
POLICIES = {
    ("openai", "eu"):       "data/policies/openai/eu_es.txt",
    ("openai", "brasil"):   "data/policies/openai/row_es.txt",
    ("openai", "row"):      "data/policies/openai/row_es.txt",
    ("anthropic", "eu"):    "data/policies/anthropic/full.txt",
    ("anthropic", "brasil"): "data/policies/anthropic/full.txt",
    ("anthropic", "row"):   "data/policies/anthropic/full.txt",
}


# --- text normalization -----------------------------------------------------
def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def normalizar(s: str) -> str:
    """Collapse whitespace, lowercase, strip accents. For robust matching."""
    s = _strip_accents(s.lower())
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def cargar_fuente_pdf(ruta: str) -> str:
    import pdfplumber
    paginas = []
    with pdfplumber.open(ruta) as pdf:
        for p in pdf.pages:
            t = p.extract_text()
            if t:
                paginas.append(t)
    return "\n".join(paginas)


def cargar_fuente_txt(ruta: str) -> str:
    return Path(ruta).read_text(encoding="utf-8")


def mejor_overlap(snippet_norm: str, fuente_norm: str) -> float:
    """
    Best contiguous matching ratio of the snippet against the source.
    SequenceMatcher over the whole source is fine for texts this size.
    """
    if not snippet_norm:
        return 0.0
    sm = SequenceMatcher(None, snippet_norm, fuente_norm, autojunk=False)
    match = sm.find_longest_match(0, len(snippet_norm), 0, len(fuente_norm))
    # Fraction of the snippet covered by its longest contiguous run in source.
    return match.size / len(snippet_norm)


def clasificar(snippet: str, fuente_raw: str, fuente_norm: str,
               umbral: float) -> tuple[str, float]:
    if not snippet or not snippet.strip():
        return "SIN_SNIPPET", 1.0
    if snippet in fuente_raw:
        return "EXACTO", 1.0
    snip_norm = normalizar(snippet)
    if snip_norm in fuente_norm:
        return "NORMALIZADO", 1.0
    ratio = mejor_overlap(snip_norm, fuente_norm)
    if ratio >= umbral:
        return "FUZZY", ratio
    return "NO_ENCONTRADO", ratio


# --- per-pipeline checks ----------------------------------------------------
def verificar_motor(umbral: float, cache: dict) -> list[dict]:
    data = json.loads(Path("scoring.json").read_text(encoding="utf-8"))
    filas = []
    for c in data["celdas"]:
        snip = c.get("motor_evidencia")
        ley = c["ley"]
        if ley not in cache:
            ruta = LEYES[ley]
            cache[ley] = cargar_fuente_pdf(ruta)
        raw = cache[ley]
        norm = cache.setdefault(ley + "::norm", normalizar(raw))
        estado, score = clasificar(snip, raw, norm, umbral)
        filas.append({
            "pipeline": "motor",
            "celda": f'{c["requisito_id"]}/{ley}',
            "articulo": c.get("motor_articulo"),
            "estado": estado,
            "score": round(score, 3),
            "snippet": snip,
        })
    return filas


def verificar_sonda(umbral: float, cache: dict) -> list[dict]:
    data = json.loads(Path("sonda.json").read_text(encoding="utf-8"))
    filas = []
    for c in data["celdas"]:
        snip = c.get("evidencia")
        key = (c["servicio"], c["locale"])
        ruta = POLICIES.get(key)
        if ruta is None:
            filas.append({
                "pipeline": "sonda",
                "celda": f'{c["requisito_id"]}/{c["servicio"]}/{c["locale"]}',
                "articulo": c.get("seccion"),
                "estado": "SIN_FUENTE_MAPEADA",
                "score": 0.0,
                "snippet": snip,
            })
            continue
        if ruta not in cache:
            cache[ruta] = cargar_fuente_txt(ruta)
        raw = cache[ruta]
        norm = cache.setdefault(ruta + "::norm", normalizar(raw))
        estado, score = clasificar(snip, raw, norm, umbral)
        filas.append({
            "pipeline": "sonda",
            "celda": f'{c["requisito_id"]}/{c["servicio"]}/{c["locale"]}',
            "articulo": c.get("seccion"),
            "estado": estado,
            "score": round(score, 3),
            "snippet": snip,
        })
    return filas


def resumen(filas: list[dict]):
    from collections import Counter
    con_snippet = [f for f in filas if f["estado"] != "SIN_SNIPPET"]
    cuenta = Counter(f["estado"] for f in filas)

    print("=" * 78)
    print("EVIDENCE-SNIPPET TRACEABILITY")
    print("=" * 78)
    print(f"Total cells:        {len(filas)}")
    print(f"Cells w/ snippet:   {len(con_snippet)}")
    for est in ("EXACTO", "NORMALIZADO", "FUZZY", "NO_ENCONTRADO",
                "SIN_FUENTE_MAPEADA", "SIN_SNIPPET"):
        if cuenta.get(est):
            print(f"  {est:20s} {cuenta[est]}")
    print()

    sospechosos = [f for f in filas
                   if f["estado"] in ("NO_ENCONTRADO", "SIN_FUENTE_MAPEADA")]
    if sospechosos:
        print("SUSPECT — verify by hand (snippet not found in source):")
        print("-" * 78)
        for f in sospechosos:
            print(f'  [{f["pipeline"]}] {f["celda"]}  (art {f["articulo"]}, '
                  f'best={f["score"]})')
            print(f'      {f["snippet"]!r}')
        print()
    else:
        print("All snippets located in their source. No manual review needed.")

    # verifiable = exact + normalized + fuzzy, out of cells that have a snippet
    ok = sum(1 for f in con_snippet
             if f["estado"] in ("EXACTO", "NORMALIZADO", "FUZZY"))
    if con_snippet:
        print(f"Verifiable snippets: {ok}/{len(con_snippet)} "
              f"({100*ok/len(con_snippet):.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--umbral", type=float, default=0.85,
                    help="fuzzy match threshold (0-1), default 0.85")
    ap.add_argument("--solo-motor", action="store_true")
    ap.add_argument("--solo-sonda", action="store_true")
    ap.add_argument("--csv", default=None, help="optional path to dump results")
    args = ap.parse_args()

    cache: dict = {}
    filas = []
    try:
        if not args.solo_sonda:
            filas += verificar_motor(args.umbral, cache)
        if not args.solo_motor:
            filas += verificar_sonda(args.umbral, cache)
    except FileNotFoundError as e:
        print(f"Source file missing: {e}", file=sys.stderr)
        print("Place the law PDFs and policy .txt files at the paths used by "
              "motor.py / sonda.py before running this check.", file=sys.stderr)
        sys.exit(1)

    resumen(filas)

    if args.csv:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["pipeline", "celda", "articulo",
                                              "estado", "score", "snippet"])
            w.writeheader()
            w.writerows(filas)
        print(f"\nDetail written to {args.csv}")


if __name__ == "__main__":
    main()
