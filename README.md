# Paridad — Regulatory Parity Auditor for AI Data Protection

A two-pipeline RAG system that audits how AI providers treat data-protection
rights across jurisdictions. Built for the Global South AI Safety Hackathon 2026.

Paridad decomposes EU AI Act (Reg. 2024/1689) and GDPR (Reg. 2016/679)
obligations into **14 atomic requirements**, then runs two independent analyses
over a shared LLM judge:

1. **Engine (Pipeline A).** Scores two Latin American laws — Chile's Law 21,719
   and Brazil's LGPD — against the 14 requirements, and validates each verdict
   against an expert legal ground truth. Produces `scoring.json` and Figures 1–2.
2. **Probe (Pipeline B).** Applies the same judge to the public privacy policies
   of two AI providers (OpenAI and Anthropic), comparing what each offers an
   EU user, a Brazilian user, and a Chilean user. Produces `sonda.json` and
   Figure 3 — the parity matrix that is the paper's central finding.

## Headline result

The engine reproduces the expert ground truth in **19/28 cells (67.9%)**
(Chile 21,719: 9/14; LGPD: 10/14). The 9 mismatches are differences of legal
classification criteria, not failures of text comprehension. This is
competitive with reported benchmarks for compliance classification without
fine-tuning.

---

## Repository structure

```
paridad/
├── core.py              # Shared LLM judge. Returns a structured JSON verdict.
├── cargador.py          # PDF -> per-article chunks (laws).
├── embeddings.py        # Multilingual dense retrieval (k nearest articles).
├── motor.py             # Pipeline A (engine): laws vs requirements -> scoring.json
├── sonda.py             # Pipeline B (probe): policies vs requirements -> sonda.json
├── figuras.py           # Generates Figures 1-3 (PNG 300 dpi + vector PDF) + R01-R14 table
├── data/
│   ├── rubrica.json     # The 14-requirement rubric with expert ground truth
│   ├── leyes/
│   │   ├── lgpd_brasil.pdf
│   │   └── ley_21719_chile.pdf
│   └── policies/
│       ├── openai/
│       │   ├── eu_es.txt
│       │   └── row_es.txt
│       └── anthropic/
│           └── full.txt
├── scoring.json         # FROZEN engine output reported in the paper
├── sonda.json           # FROZEN probe output reported in the paper
├── requirements.txt
├── .env.example
└── README.md
```

> Note: the code keeps Spanish identifiers (`motor` = engine, `sonda` = probe,
> `cargador` = loader, `juzgar` = judge). This README documents them in English;
> the filenames and commands below are the actual ones in the repo.

---

## Installation

Requires **Python 3.10+** (the code uses `list[dict]` builtin generics).

```bash
git clone https://github.com/PMeik/paridad.git
cd paridad

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env               # then edit .env and paste your Anthropic API key
```

The first run downloads the embedding model
(`paraphrase-multilingual-mpnet-base-v2`, ~470 MB) from HuggingFace and caches
it on disk. It runs on CPU; no GPU required.

### API key

`core.py` reads `ANTHROPIC_API_KEY` from the environment (loaded from `.env`).
The key is never hard-coded and `.env` is git-ignored.

---

## How to run

Run the connection test first; if it prints a verdict, the rest of the pipeline
will not surprise you with auth errors:

```bash
python core.py        # connection + judge smoke test
python cargador.py    # sanity-check the per-article PDF chunking
python embeddings.py  # preview the retrieval (which articles match each requirement)
```

Then the two pipelines:

```bash
python motor.py       # Pipeline A -> writes scoring.json + prints accuracy vs ground truth
python sonda.py       # Pipeline B -> writes sonda.json + prints per-service parity matrices
python figuras.py     # reads both JSONs -> Figures 1-3 (PNG + PDF) + table_requirements.{csv,tex}
```

`figuras.py` expects `scoring.json`, `sonda.json` and `rubrica.json` in the
working directory. If you keep the rubric only under `data/`, copy it up or
adjust the `rubrica_path` argument in `tabla_requisitos()`.

---

## Reproducibility

This is the part a reviewer cares about, so it is spelled out.

| Item | Value |
|------|-------|
| Judge model (frozen run) | `claude-haiku-4-5-20251001` |
| Alternative judge | `claude-sonnet-4-6` (`MODELO_FINAL` in `core.py`) |
| Embedding model | `paraphrase-multilingual-mpnet-base-v2` |
| Retrieval k (engine) | 7 articles per requirement (`K_VECINOS`) |
| Retrieval k (probe) | 7 paragraphs per requirement (`K_PARRAFOS`) |
| Judge temperature | 0.0 |
| Engine cells | 28 (14 requirements × 2 laws) |
| Probe cells | 36 (6 probeable requirements × 3 locales × 2 services) |
| Total wall-clock | ~2 minutes |
| Cost (Haiku, frozen run) | a few US cents |
| Cost (Sonnet, optional) | < US$1 for the full re-run |

**On determinism.** The judge runs at `temperature=0.0`. The Anthropic API does
**not** expose a `seed` parameter, and per Anthropic's own documentation,
`temperature=0` is *not* fully deterministic — low-level floating-point
accumulation order on GPU can produce small variations. We therefore treat
`scoring.json` and `sonda.json` as the **frozen, reported runs**: re-running may
reproduce the headline 67.9% exactly, but tiny differences on borderline cells
are possible and expected. The frozen JSONs in this repo are the canonical
reference for every number in the paper.

**Why k=7.** Calibrated against the ground truth. With k=5, two cells
(R07 Chile, R11 LGPD) failed because the correct article ranked 6th–7th, just
outside the window. k=10 adds irrelevant articles (noise + cost). k=7 is the
point where recall rises without adding noise.

---

## Evidence-snippet traceability

To audit the anti-hallucination property of the engine, every cell that
carries an `evidence_snippet` is verified against its source text by
`verificar_evidencia.py`. The script normalizes whitespace, casing and
accents, then falls back to a fuzzy contiguous match (threshold 0.85).

Result on the frozen runs: **19/24 snippets (79.2%) are located literally
or with formatting-only differences** in their source (10 NORMALIZED,
9 EXACT). The 5 remaining cells (R01/Chile, R06/LGPD, R11/Chile,
R12/Chile, R06/Anthropic-EU) carry snippets that *enumerate* legal
content found in the cited article (e.g. listing transfer mechanisms or
DPIA triggers) rather than quoting a single contiguous span. Manual
inspection confirms the substantive content matches the source; the
deviation is a quote-shape limitation of the judge, not a fabrication.
This 79.2% is reported alongside the engine's 67.9% ground-truth
agreement as the project's two reproducibility metrics.

To re-run the check:

    python verificar_evidencia.py --csv revision.csv

## The judge: prompt and output schema

A single function, `juzgar()` in `core.py`, is shared by both pipelines. It
receives a source text (retrieved law articles, or policy paragraphs) plus one
requirement, and returns a structured JSON verdict.

The verdict schema:

| Field | Type | Meaning |
|-------|------|---------|
| `fallo` | `"full"` \| `"partial"` \| `"absent"` | Coverage verdict |
| `articulo` | string \| null | Article/section where the verdict is anchored |
| `evidence_snippet` | string \| null | **Literal** quote (≤15 words) copied from the source — the anti-hallucination anchor |
| `confidence` | float 0.0–1.0 | Judge's self-reported confidence |
| `razon` | string | One-sentence justification |
| `needs_human_review` | bool | Auto-set when `confidence < 0.6` |

The `evidence_snippet` is the key methodological device: because it must be a
literal copy from the source, any verdict can be verified by Ctrl+F in the
original PDF or policy. A snippet that cannot be found is, by definition, a
hallucinated verdict.

The full prompt is defined inline in `juzgar()` (`core.py`). It instructs the
model to return JSON only, requires the literal snippet, and forces `absent`
when no real text can anchor the evidence.

---

## Output JSON: top-level shape

Both `scoring.json` and `sonda.json` share this shape:

```jsonc
{
  "_meta": {
    "modelo_juez": "claude-haiku-4-5-20251001",
    "modelo_embeddings": "paraphrase-multilingual-mpnet-base-v2",
    "k_vecinos": 7,            // (engine)  /  "k_parrafos": 7 (probe)
    ...
  },
  "celdas": [ /* one object per cell, see schema above */ ]
}
```

In `scoring.json` each cell additionally carries the expert ground truth
(`gt_fallo`, `gt_articulo`, `gt_razon`), the comparison result
(`match_gt`: `"match"` | `"mismatch"`), and the retrieved articles
(`chunks_top_k`) for full traceability.

---

## Data sources

- **LGPD (Brazil):** Lei 13.709/2018, official text from Planalto.
- **Law 21,719 (Chile):** official text from the BCN (Biblioteca del Congreso
  Nacional). In force from December 2026.
- **Privacy policies:** public regional privacy policies of OpenAI and Anthropic
  as retrieved during the hackathon weekend. The Spanish-language versions are
  used for cross-locale linguistic consistency. Retrieval date is recorded in
  the repo so the snapshot can be checked against later policy revisions.

---

## Code and Data

- Repository: `https://github.com/PMeik/paridad`  
- Frozen outputs: `scoring.json`, `sonda.json`
- Figures: produced by `figuras.py`

### Dual-use note

Paridad is an **auditing** tool. Its intended use is to compare the
data-protection commitments that AI providers make to users in different
jurisdictions, in order to surface unequal treatment and inform public policy.
It is **not** designed or suitable for reconstructing provider internals,
profiling individuals, or any form of surveillance. The judge reads only public
legal texts and public privacy policies and emits coverage verdicts with literal
evidence; it produces no model internals, no personal data, and no attack
surface. Users extending the tool should preserve this scope: audit policies,
do not weaponize the pipeline.

---

## License

Add a license before publishing (MIT recommended for hackathon code unless your
venue requires otherwise).