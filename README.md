# Agriculture Schemes RAG

A retrieval system over Government of India agriculture scheme documents,
with a structured eligibility layer and RAG over scheme prose. Built with
rigorous evaluation and ablation.

V1 covers seven schemes — PM-KISAN, PMFBY, KCC, SMAM, MIDH, NFSM, AIF —
selected for knowledge diversity across income support, crop insurance,
agricultural credit, farm mechanization, horticulture, food security, and
infrastructure financing.

Three layers, routed deterministically:

1. **Structured** — attribute-based eligibility as a database query.
2. **Workflow** — portal step-sequences served deterministically from
   curated CSVs when a question maps to a known workflow.
3. **RAG** — retrieval over authoritative prose (procedures, definitions,
   exceptions, amendments, cross-scheme interactions), with citations.

---

Full write-up — problem framing, architecture diagram, ablation table,
how to run — lands in the write-up phase. See `scope.md` for the
authoritative scope document, `CLAUDE.md` for the project brief, and
`DECISIONS.md` for the running design log.
