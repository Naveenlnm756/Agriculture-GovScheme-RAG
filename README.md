# Agriculture Schemes RAG

A retrieval system over Government of India agriculture-scheme documents.
A user (farmer, extension officer, policy analyst) asks a plain-English
question — *"What is the current subsidy rate on a rotavator under SMAM,
and which guideline set it?"* — and the system returns the current answer
with the exact source document quoted as proof.

**Live demo:** https://agri-schemes-rag-production.up.railway.app/

---

## The problem, in one paragraph

V1 covers **seven schemes** issued by the Department of Agriculture &
Farmers Welfare (DA&FW), with RBI as the regulatory source for the
credit scheme (KCC): **PM-KISAN, PMFBY, KCC, SMAM, MIDH, NFSM, AIF**.
The corpus spans ~10+ years of amendments. Individual schemes have
deep temporal chains where later guidelines supersede earlier ones,
and many questions are multi-hop and cross-document — *"what was the
subsidy rate for a rotavator in 2018 vs 2025?"* cannot be answered
from any single PDF. Every answer must cite its source; an unsourced
answer is worthless in a scheme-eligibility context.

## Architecture: two layers (V1)

`scope.md` §3 lays out a three-layer design; V1 ships two of them.

1. **Structured eligibility layer.** Deterministic filter over farmer
   attributes → list of schemes that are *potentially applicable*. No
   LLM on the routing path. Rules live in
   [`data/scheme_eligibility.json`](data/scheme_eligibility.json)
   with per-rule provenance — every applicability / disqualifying /
   boost rule carries `{filename, page, note}` so the structured layer
   is as citable as the RAG layer. Language contract: says *"potentially
   applicable"*, never *"eligible"*.

2. **RAG layer.** Retrieval over authoritative prose — definitions,
   conditions, exceptions, cross-provision interactions, "what changed
   in 2018", "which document supersedes this rule". Pipeline:

   ```
   query
     → hybrid retrieval  (semantic top-20 + BM25 top-20, RRF-fused k=60)
     → cross-encoder rerank  (ms-marco-MiniLM → top-5)
     → confidence gate  (refuse if top-1 rerank score < -2.0)
     → grounded generation with citations
   ```

3. *(Workflow layer, deferred.)* Portal SOP CSVs live under
   `data/raw/[SCHEME]/02_Workflows/` and were the planned third layer.
   Deferred to V2 because the eval showed RAG handles the
   `simple_procedural` category at hit@5=0.92 without it — a
   deterministic router would duplicate what's working. Full reasoning
   in `DECISIONS.md`.

## The ablation table

Every technique was measured as an isolated row against the frozen
text-only anchor collection (`agri_schemes_rag`, n=78 golden set).
Each row applies **one** intervention on top of the baseline — not
stacked — so the metric delta attributes to a specific technique.

| # | Row | hit@5 | hit@10 | MRR | Faithfulness | Ans. Rel. | Ctx. Prec. | Ctx. Rec. |
|---|---|---|---|---|---|---|---|---|
| 0 | Baseline (semantic-only) | 0.628 | 0.718 | 0.459 | 0.694 | 0.635 | 0.746 | 0.642 |
| 1 | + Cross-encoder rerank | **0.692** | 0.692 | 0.479 | **0.749** | **0.724** | **0.831** | 0.629 |
| 2 | + Hybrid (BM25 + RRF) | 0.654 | 0.654 | 0.475 | 0.729 | 0.733 | 0.744 | 0.618 |
| 3 | + Query transform (multi-query) | 0.654 | 0.654 | 0.440 | 0.637 | 0.611 | 0.732 | 0.602 |

**How to read this:**

- **Reranker (row 1) is the biggest single win.** +6.4 pts hit@5,
  +5.5 pts faithfulness, +8.5 pts context precision. The reranker's
  job is to promote the correct chunk into the top-5 the generator
  sees, and it earns that promotion.
- **Hybrid (row 2) is a modest positive.** Faithfulness and answer
  relevancy both up. Context precision slightly down vs baseline —
  BM25 pulls in more candidates, some of which are lexically similar
  but topically off. The net is still positive at the answer level.
- **Query transform (row 3) is a documented net negative.** Multi-query
  fusion added ranked lists that reordered the top-5 in ways that hurt
  answer quality on this corpus. Kept in the table because negative
  results are results — hiding them would make the ablation dishonest.
  Detailed diagnosis in `DECISIONS.md` (2026-09-14 entry).
- **hit@10 == hit@5 in rows 1-3** because those rows return exactly 5
  chunks to the generator. Only the baseline row has a 20-deep pool
  to inspect at rank 10. The reranker's / hybrid's / query-transform's
  job *is* to produce a top-5 — if the correct chunk isn't in it, the
  technique has failed regardless of what the deeper pool contained.

## Headline row: the shipped stack

The production pipeline combines the winning techniques and runs
against the **multimodal collection** (text + tables + images from
OCR / vision-model ingestion). Measured end-to-end on the full n=88
golden set:

| Metric | Value | n |
|---|---|---|
| hit@5 | 0.671 | 88 |
| MRR | 0.499 | 88 |
| Faithfulness | 0.726 | 80 |
| Answer relevancy | 0.690 | 80 |
| Context precision | 0.801 | 80 |
| Context recall | 0.697 | 80 |
| OOS refusal correctness | **1.000** | 8 |

Full artefacts: [`eval/results/20260920-214207_full_n88/`](eval/results/).

**Per-category strengths and weaknesses:**

| Category | n | hit@5 | Faith. | Note |
|---|---|---|---|---|
| simple_procedural | 12 | 0.917 | 0.917 | Strongest — portal SOP questions |
| late_content_diagnostic | 8 | 0.875 | 0.781 | Late-added hard questions retrieve well |
| definition | 8 | 0.750 | 0.986 | Grounded definitions near ceiling |
| simple_factual | 10 | 0.700 | 0.867 | Reliable on single-doc facts |
| temporal_supersession | 10 | 0.700 | 0.800 | Ctx recall only 0.35 — surfaces *a* relevant doc, not always the superseding one |
| cross_scheme | 10 | 0.600 | **0.275** | Retrieval half-works; generator hallucinates on mixed-scheme context |
| multi_hop_scheme | 12 | 0.583 | 0.667 | MRR 0.28 — correct chunk in pool, not near top |
| out_of_scope | 8 | — | — | **8/8 refused correctly** |

**Honest ceiling:** 29/88 questions miss at hit@10 (33% retrieval-side
failure rate). Anything that beats 0.67 hit@10 on this golden set is a
real gain.

## Design principles that shaped this project

- **Measurement is the product.** Every technique is justified by a
  measured failure of the previous state, not by "it's the fashionable
  thing to add." The query-transform row is kept in the ablation table
  even though it's negative — it's evidence, not marketing.
- **Explicit over clever.** The retrieval pipeline is written as a
  visible sequence of stages (retrieve → rerank → generate), not
  hidden behind an orchestration framework. Every module in
  [`src/retrieval/`](src/retrieval) is one page of code with one job.
- **Ablation-integrity rules.** Generator model, judge model, prompt
  temperature, and reasoning effort are held constant across every
  row of a single ablation table. A metric movement must attribute to
  the technique under study — not to a model swap. Documented in
  [`CLAUDE.md`](CLAUDE.md) §11 and enforced in `src/config.py`.
- **Judge independence.** RAGAS uses `mistral-small-2603` as the
  judge — a different provider from the generator (`openai/gpt-oss-20b`
  on Groq). Same-model self-judging inflates faithfulness by
  ~10 points in the literature; keeping the judge independent removes
  the bias.

## Stack

- Retrieval: `chromadb` + `sentence-transformers` (`BAAI/bge-small-en-v1.5`) + `rank-bm25`
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Vision (ingestion): `gemini-3.5-flash-lite` — image chunks for tables, diagrams, scanned pages
- Generation: `openai/gpt-oss-20b` on Groq (locked for the whole ablation)
- Evaluation: `ragas` with an independent Mistral judge
- API: `fastapi` + `uvicorn`, streamed responses, per-request latency + token logging
- UI: `streamlit`
- Persistence: `sqlite` for chat history (kept separate from Chroma —
  chat lifecycle is orthogonal to corpus rebuilds)
- Hosting: Railway

## Repo layout

```
├── data/
│   ├── raw/[SCHEME]/{01_RAW_PDFs,02_Workflows}/    # curated corpus, hand-collected
│   ├── processed/                                  # chunked + embedded output
│   ├── chroma_db/                                  # ablation anchor (text-only)
│   ├── chroma_prod/                                # multimodal production store
│   └── scheme_eligibility.json                     # structured layer (v2 schema, per-rule sources)
├── src/
│   ├── config.py                # all knobs, models, thresholds, feature flags
│   ├── ingestion/               # PDF + workflow-CSV loaders, chunkers
│   ├── embeddings/              # text → vectors
│   ├── retrieval/               # semantic + BM25 + hybrid + reranker
│   ├── vision/                  # image-chunk vision pipeline
│   ├── query/                   # multi-query rewriter (ablation row #3 only)
│   ├── production/              # scheme_discovery + confidence gate + composed pipeline
│   ├── generation/              # grounded generator with citations
│   ├── chat/                    # SQLite conversation store (rendering only)
│   ├── eval/                    # rotating Groq client, Mistral judge wrapper
│   └── api/                     # FastAPI endpoint, streaming, logging
├── eval/
│   ├── golden_set.json                     # 88 hand-built Q/A/source triples
│   ├── run_eval.py                         # golden run + RAGAS + retrieval metrics
│   └── results/                            # per-run artefacts (jsonl + summary)
├── scope.md          # authoritative scope document (V1.1)
├── CLAUDE.md         # project brief and collaboration rules
├── DECISIONS.md      # design log — one entry per real choice
└── README.md         # this file
```

## Running locally

```powershell
# Python 3.11 (locked — see CLAUDE.md §12)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# .env must set at least:
#   GROQ_API_KEY=...                (generator)
#   GROQ_API_KEYS=...               (comma-separated pool for eval)
#   MISTRAL_API_KEY=... or MISTRAL_API_KEYS=...   (judge)
#   GEMINI_API_KEY=...              (vision ingestion — optional if not re-ingesting)

# Build the production collection (multimodal — text + tables + images)
python scripts/build_prod_collection.py

# Run the FastAPI server
uvicorn src.api.main:app --reload --port 8000

# Run the Streamlit UI (separate terminal)
streamlit run streamlit_app.py
```

## Running the eval

```powershell
# Baseline row (semantic-only, text anchor collection)
python eval/run_eval.py --pipeline_mode baseline --full --sleep 2.0

# Ablation rows
python eval/run_eval.py --pipeline_mode reranked --full
python eval/run_eval.py --pipeline_mode hybrid --full
python eval/run_eval.py --pipeline_mode query_transform --full

# Headline row (shipped stack on multimodal collection)
python eval/run_eval.py --pipeline_mode full --full --sleep 2.0
```

Each run writes to `eval/results/{timestamp}_{mode}_n{count}/`:
- `results.jsonl` — one line per question (retrieval + generation + RAGAS)
- `progress.json` — completion snapshot (supports `--resume`)
- `summary.json` — aggregate + per-category breakdown

## Design docs

- [`scope.md`](scope.md) — authoritative V1 scope: what's in, what's
  out, known limitations, why the corpus is chosen the way it is.
- [`DECISIONS.md`](DECISIONS.md) — running design log. One entry per
  real choice, with the measured effect where applicable.
