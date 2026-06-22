"""
figuras.py — generates the paper figures (English, camera-ready).

FIGURE 1 (gap map):       color = expert ground truth. The real legal gap.
FIGURE 2 (engine valid.): color = engine verdict + checkmark where it matches
                          ground truth. Shows how well the automated engine
                          reproduces the expert analysis.
FIGURE 3 (parity matrix): color = verdict from the probe over AI privacy
                          policies. The central finding: read rows within each
                          service to see the EU > Brazil > Chile gradient.

Each figure is exported as both high-resolution PNG (300 dpi) and vector PDF.
PNG goes into the .docx; PDF is preferred for LaTeX/print.

Run:  python figuras.py
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

# DejaVu Sans ships with matplotlib and contains the checkmark glyph (✓),
# so annotations render reliably across backends.
plt.rcParams["font.family"] = "DejaVu Sans"


# Consistent verdict palette across all figures.
COLORES = {
    "full":                   "#2E7D32",  # green
    "partial":                "#F9A825",  # amber
    "absent":                 "#C62828",  # red
    "absent_sin_enforcement": "#8E24AA",  # purple
    "error_parseo":           "#9E9E9E",  # grey
    "error_api":              "#9E9E9E",
}

ORDEN_FALLO = ["full", "partial", "absent", "absent_sin_enforcement"]

# Shared colormap: -1=grey, 0=purple, 1=red, 2=amber, 3=green
CMAP = ListedColormap(["#9E9E9E", "#8E24AA", "#C62828", "#F9A825", "#2E7D32"])
BOUNDS = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
NORM = BoundaryNorm(BOUNDS, CMAP.N)


def fallo_a_valor(fallo: str) -> int:
    """Map a verdict string to an integer used as a color index."""
    mapeo = {
        "full":                   3,
        "partial":                2,
        "absent":                 1,
        "absent_sin_enforcement": 0,
    }
    return mapeo.get(fallo, -1)


def _guardar(fig_base: str):
    """Save the current figure as 300-dpi PNG and vector PDF."""
    plt.savefig(fig_base + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(fig_base + ".pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_base}.png  +  {fig_base}.pdf")


# ---------------------------------------------------------------------------
# FIGURE 1 / 2 — engine heatmap (gap map and engine validation)
# ---------------------------------------------------------------------------
def figura_motor(scoring_path: str = "scoring.json",
                 salida: str = "figure1b_engine",
                 modo: str = "motor"):
    """
    modo="gt"    -> color = expert ground truth          (Figure 1, real gap)
    modo="motor" -> color = engine verdict + match marks (Figure 2, validation)
    """
    with open(scoring_path, encoding="utf-8") as f:
        data = json.load(f)
    celdas = data["celdas"]

    reqs = sorted({c["requisito_id"] for c in celdas})   # R01..R14
    leyes = sorted({c["ley"] for c in celdas})

    matriz = np.zeros((len(leyes), len(reqs)))
    matches = np.zeros((len(leyes), len(reqs)), dtype=bool)
    for c in celdas:
        i = leyes.index(c["ley"])
        j = reqs.index(c["requisito_id"])
        fallo = c["gt_fallo"] if modo == "gt" else c["motor_fallo"]
        matriz[i, j] = fallo_a_valor(fallo)
        matches[i, j] = (c["match_gt"] == "match")

    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.imshow(matriz, cmap=CMAP, norm=NORM, aspect="auto")

    ax.set_xticks(range(len(reqs)))
    ax.set_xticklabels(reqs, fontsize=10)
    ax.set_yticks(range(len(leyes)))
    ax.set_yticklabels(
        [{"Chile21719": "Chile (Law 21,719)",
          "LGPD":       "Brazil (LGPD)"}.get(l, l) for l in leyes],
        fontsize=11,
    )

    # Checkmarks only in validation mode (meaningless on the ground-truth map).
    if modo == "motor":
        for i in range(len(leyes)):
            for j in range(len(reqs)):
                if matches[i, j]:
                    ax.text(j, i, "\u2713", ha="center", va="center",
                            color="white", fontsize=11, fontweight="bold")

    titulo = {
        "gt": ("Figure 1 — Regulatory gap map\n"
               "Coverage of EU AI Act + GDPR requirements by each "
               "Latin American law\n"
               "(expert ground-truth assessment)"),
        "motor": ("Figure 2 — Automated engine validation\n"
                  "Engine verdict per requirement vs. expert ground truth\n"
                  "(\u2713 = engine matches expert assessment)"),
    }[modo]
    ax.set_title(titulo, fontsize=11, pad=12)

    handles = [
        mpatches.Patch(color="#2E7D32", label="Full"),
        mpatches.Patch(color="#F9A825", label="Partial"),
        mpatches.Patch(color="#C62828", label="Absent"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=3, frameon=False, fontsize=10)

    plt.tight_layout()
    _guardar(salida)


# ---------------------------------------------------------------------------
# FIGURE 3 — probe parity matrix (central finding)
# ---------------------------------------------------------------------------
def figura_sonda(sonda_path: str = "sonda.json",
                 salida: str = "figure3_parity"):
    with open(sonda_path, encoding="utf-8") as f:
        data = json.load(f)
    celdas = data["celdas"]

    reqs = sorted({c["requisito_id"] for c in celdas})

    # Rows = (service, locale), grouped by service, locales EU -> Brazil -> Chile.
    orden_locales = ["eu", "brasil", "row"]
    servicios = sorted({c["servicio"] for c in celdas})
    filas = []
    for serv in servicios:
        for loc in orden_locales:
            if any(c["servicio"] == serv and c["locale"] == loc for c in celdas):
                filas.append((serv, loc))

    matriz = np.full((len(filas), len(reqs)), -1.0)
    for c in celdas:
        try:
            i = filas.index((c["servicio"], c["locale"]))
            j = reqs.index(c["requisito_id"])
            matriz[i, j] = fallo_a_valor(c["fallo"])
        except ValueError:
            pass

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.imshow(matriz, cmap=CMAP, norm=NORM, aspect="auto")

    ax.set_xticks(range(len(reqs)))
    ax.set_xticklabels(reqs, fontsize=10)

    nombre_serv = {"anthropic": "Anthropic", "openai": "OpenAI"}
    etiquetas_locale = {
        "eu":     "EU/EEA",
        "brasil": "Brazil (LGPD)",
        "row":    "Chile (Rest-of-World tier)",
    }
    ax.set_yticks(range(len(filas)))
    ax.set_yticklabels(
        [f"{nombre_serv.get(serv, serv)} \u00b7 {etiquetas_locale[loc]}"
         for serv, loc in filas],
        fontsize=10,
    )

    # Horizontal line separating services.
    for k in range(1, len(servicios)):
        ax.axhline(y=k * len(orden_locales) - 0.5, color="black", linewidth=1.2)

    ax.set_title(
        "Figure 3 — Data-protection parity matrix\n"
        "What AI privacy policies offer the user by jurisdiction\n"
        "(compare rows within each service for the EU > Brazil > Chile gradient)",
        fontsize=11, pad=12,
    )

    handles = [
        mpatches.Patch(color="#2E7D32", label="Offers (full)"),
        mpatches.Patch(color="#F9A825", label="Partial"),
        mpatches.Patch(color="#C62828", label="Does not offer (absent)"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=3, frameon=False, fontsize=10)

    plt.tight_layout()
    _guardar(salida)


# ---------------------------------------------------------------------------
# Requirement legend table R01..R14 (makes the figures self-explanatory)
# ---------------------------------------------------------------------------
def tabla_requisitos(rubrica_path: str = None,
                     salida_csv: str = "table_requirements.csv",
                     salida_tex: str = "table_requirements.tex"):
    """
    Emit the R0X -> requirement mapping as CSV (for the .docx) and as a LaTeX
    longtable snippet (for the IEEE template). Short English glosses are written
    by hand so the table is compact; the full Spanish text stays in rubrica.json.

    Looks for the rubric in the current dir and under data/ so the script runs
    regardless of where rubrica.json lives.
    """
    import os
    if rubrica_path is None:
        for cand in ("rubrica.json", "data/rubrica.json"):
            if os.path.exists(cand):
                rubrica_path = cand
                break
        else:
            raise FileNotFoundError(
                "rubrica.json not found in '.' or 'data/'. "
                "Pass rubrica_path explicitly."
            )
    with open(rubrica_path, encoding="utf-8") as f:
        rub = json.load(f)

    # Compact English glosses, one per requirement, kept short for the appendix.
    glosas = {
        "R01": "Prohibited AI practices (manipulation, social scoring, mass biometric ID)",
        "R02": "Credit scoring classified as high-risk AI",
        "R03": "Right not to be subject to solely automated decisions / profiling",
        "R04": "Guaranteed meaningful human intervention in automated decisions",
        "R05": "Right to meaningful information on the logic of automated processing",
        "R06": "International data transfers governed by adequacy / safeguards / BCR",
        "R07": "Foreign companies must appoint a local representative with own liability",
        "R08": "GPAI providers must publish a training-data content summary",
        "R09": "Legal definition of deepfake (AI-manipulated synthetic content)",
        "R10": "Generative-AI output marked as synthetic in machine-readable form",
        "R11": "Biometric data classified as a special category of sensitive data",
        "R12": "Mandatory Data Protection Impact Assessment for high-risk processing",
        "R13": "Data protection by design and by default",
        "R14": "Collective exercise of data-subject rights (class action)",
    }

    filas = []
    for r in rub["requisitos"]:
        rid = r["id"]
        dim = r["dimension"]
        capa = r["capa"]
        std = r["articulo_estandar"]
        filas.append((rid, glosas.get(rid, r["requisito"][:80]), dim, capa, std))

    # CSV
    import csv
    with open(salida_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Requirement", "Dimension", "Layer", "EU standard"])
        w.writerows(filas)

    # LaTeX longtable (IEEE-friendly)
    lineas = [
        r"\begin{table}[t]",
        r"\caption{Requirement legend (R01--R14). Layer A = EU AI Act, "
        r"B = GDPR.}",
        r"\label{tab:requirements}",
        r"\centering\footnotesize",
        r"\begin{tabular}{@{}llp{4.6cm}c@{}}",
        r"\toprule",
        r"ID & Layer & Requirement & EU std. \\",
        r"\midrule",
    ]
    for rid, glosa, dim, capa, std in filas:
        g = glosa.replace("&", r"\&")
        s = std.replace("&", r"\&")
        lineas.append(f"{rid} & {capa} & {g} & {s} \\\\")
    lineas += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    with open(salida_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas) + "\n")

    print(f"Saved: {salida_csv}  +  {salida_tex}")


if __name__ == "__main__":
    figura_motor(salida="figure1a_gap",    modo="gt")     # Figure 1
    figura_motor(salida="figure2_engine",  modo="motor")  # Figure 2
    figura_sonda(salida="figure3_parity")                 # Figure 3
    tabla_requisitos()                                    # R01..R14 legend
    print("\nDone. PNGs go into the .docx; PDFs into the LaTeX template.")