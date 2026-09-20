# Agriculture Schemes RAG — Project Scope

**Version 1.1 — locked after Phase 1 collection.**
Supersedes V1.0 which was written pre-collection.
Revisions require a new version number and a corresponding DECISIONS.md entry.

---

## 1. Purpose

The project builds a retrieval system over Government of India agriculture
scheme documents. It is designed to answer citation-grounded questions about
eligibility, procedures, definitions, exceptions, financial rules and
scheme-to-scheme interactions.

The corpus is bounded and authoritative rather than large. The design
prioritises regulatory relevance, source authority, temporal coverage and
traceability over raw document count.

The system must support:

- accurate retrieval of the applicable scheme rules;
- historical and current questions;
- amendment and supersession reasoning;
- cross-document and cross-scheme regulatory reasoning;
- multi-hop questions across schemes and layers;
- construction of a manually validated golden evaluation set;
- deterministic answers where the question is deterministic (eligibility,
  known workflows) and retrieved answers where the question is prose-bound
  (procedures, exceptions, definitions, interactions).

---

## 2. Schemes in Scope (V1)

The V1 corpus is fixed at the following seven schemes.

| Scheme | Category | Why in scope (evidence from collection) |
|---|---|---|
| **PM-KISAN** | Income Support | Clean base case for farmer eligibility, exclusions, beneficiary identification, DBT and scheme exceptions. Real prose from Operational Guidelines, both original and Revised FAQs (enables temporal reasoning), Amendment to OG, refund mechanism, Aadhaar seeding, physical verification, NPCI transfer, e-KYC. Portal workflows captured for registration, e-KYC (three methods: OTP, biometric, face), grievance, refund, address correction, voluntary surrender. |
| **PMFBY** | Crop Insurance | Richest procedural depth in the V1 corpus. Three versions of Operational Guidelines (Revamped 2020, Revised, and one 213-page edition), both original 2016 Notifications, Crop Calendar, WINDS and YESTECH manuals, penalty clause invocation, subsidy-through-challan and premium-routing letters. Enables genuine multi-hop across notified crops, notified areas, risks, premium computation, claim procedures, loss assessment and timelines. |
| **KCC** | Agricultural Credit | Textbook regulatory supersession chain: RBI Master Circular 2017 → RBI Circular 2022 → four bank-type-specific RBI Directions in 2026 (Commercial Banks, Regional Rural Banks, Rural Cooperative Banks, Small Finance Banks). The four 2026 Directions share the same date but differ by implementer, enabling cross-document multi-hop questions ("does the KCC limit differ by bank type in 2026?"). AHDF-KCC variant and MISS interest-subvention scheme included as KCC-adjacent (see scope decisions below). |
| **SMAM** | Farm Mechanization | Deepest temporal chain in the corpus: six versions of Guidelines spanning 2016-17 → 2018-19 → 2019 revised → 2020-21 → 2024 → 2025 revised (Dec 18). Supports "what was the subsidy rate for a rotavator in 2018 vs 2025" type questions. Two workflows captured (Single Implement Subsidy, Custom Hiring Centre Establishment) — genuinely distinct procedures, not padded variants. |
| **MIDH** | Horticulture Development | Adds the most sophisticated procedural knowledge in the corpus: NHB Subsidy Claim workflow (12 steps involving estimated vs final claim routes, Format II-A/II-B, Subsidy Reserve Fund Account, Joint Inspection Team, 18-month completion window). Six NHB workflows total. Also brings NCCD Engineering Guidelines (395 pages) and NCCD Basic Datasheets — cold-chain technical standards that cross-reference AIF-funded infrastructure. |
| **NFSM** | Crop Productivity / Food Security | Cleanest scan situation (zero scanned PDFs). Deepest text-extractable temporal chain (2009 → 2013 → 12th Five Year Plan → 2018 Revamped → 2018-19/2019-20 combined → 2025-26). Workflows uniquely captured at state level (Maharashtra via MahaDBT, Rajasthan, Meghalaya via plain-paper application, Tamil Nadu), reflecting the reality that NFSM is centrally-designed but state-implemented. |
| **AIF** | Agriculture Infrastructure Financing | The only *financing-shape* scheme in the corpus — every other scheme is a subsidy/support scheme, AIF is credit-linked project financing with interest subvention and credit guarantee. Distinct regulatory shape (eligible project types, interest subvention up to ₹2 crore, CGTMSE/NAB Sanrakshan credit guarantee, term-loan structure). Lending-Institution-facing workflows (Interest Subvention Claim, CGTMSE Fee Claim) correctly reflect that AIF applications go through banks, not a farmer portal. |

### Scope decisions on adjacent/variant schemes

The following adjacent schemes were considered during collection and their
inclusion status is explicit.

- **PM-KMY (Pradhan Mantri Kisan Maan-Dhan Yojana)** — pension scheme for
  the same beneficiary population as PM-KISAN. **Included** as a related
  scheme under the PM-KISAN folder because it enables cross-scheme
  interaction questions and shares the beneficiary base. Not a silent
  addition — recorded here.

- **MISS (Modified Interest Subvention Scheme)** — interest subvention
  mechanism for KCC loans. **Included** as KCC-adjacent because interest
  subvention directly affects what a KCC borrower pays and is operationally
  part of the KCC picture.

- **AHDF-KCC (Animal Husbandry, Dairying, Fisheries KCC variant)** —
  **Included** as a form of KCC. Distinct enough from the primary crop-loan
  KCC to justify a separate document but part of the KCC family.

- **NMEO-Oil Palm** — successor to NFSM's oilseed sub-mission after
  reorganisation. **Included** under NFSM because it is the operational
  continuation of the NFSM oilseed component.

### V1 explicit exclusions

- **RWBCIS (Restructured Weather Based Crop Insurance Scheme)** — excluded
  because its knowledge substantially overlaps PMFBY without providing
  distinct domain diversity. A joint PMFBY/WBCIS administrative order was
  retained where the PMFBY substance justified inclusion.
- Any general RBI, MSME or priority-sector-lending documents that mention
  AIF only in passing are excluded to prevent scope creep.
- General agricultural policy documents, budget announcements and unrelated
  ministry material are excluded regardless of tangential connection.

### Scope boundary

The V1 corpus is fixed at these seven schemes plus the adjacent inclusions
listed above. An eighth primary scheme will only be added in a later version
if it introduces a genuinely distinct body of knowledge rather than merely
increasing corpus size.

---

## 3. Three Layers: Structured, RAG, and Workflow

Phase 1 collection revealed that the project has **three** distinct
knowledge layers, not two. The original V1.0 scope described only the
structured/RAG split; the workflow layer is added as a first-class layer
in V1.1.

### Structured Layer

Deterministic filtering over explicit farmer/entity attributes.

Purpose: answer *"Based on the user's known attributes, which schemes
are potentially relevant?"* — as a database query, not a retrieval.

Typical fields (scheme-aware, not a universal form):

- Scheme; beneficiary/farmer type
- State / geographic scope; season / notified-area
- Age; landholding size; ownership status
- Social category; income/economic status
- Crop / activity; machinery / infrastructure / project type
- Loan / financing requirement
- Benefit type; benefit amount / assistance rate

### RAG Layer

Retrieval over authoritative prose.

Purpose: answer questions that require interpreting scheme prose — procedures,
definitions, conditions, exceptions, benefits, claim mechanics, implementation
rules, amendments, interactions between provisions.

Content and metadata preserved:

- Scheme; document title; document type; version / date
- Section / subsection; page number
- Issuing authority; source URL
- Amendment / supersession relationship where identifiable

### Workflow Layer

Step-by-step procedural knowledge extracted from official scheme portals
and operational SOPs, stored as structured CSVs.

Each workflow row carries: `workflow_id, step_no, instruction,
input_required, condition, source_title, official_url`.

The workflow layer captures knowledge that is procedural and deterministic
but is **not in the prose PDFs** — it lives on the portals or in
step-by-step SOPs. Serving these deterministically is faster, more
accurate and more honest than embedding them into RAG and retrieving
approximate answers.

Examples of workflows captured in V1:

- PM-KISAN: registration, e-KYC (OTP/biometric/face), grievance, refund,
  address correction, voluntary surrender
- SMAM: single-implement subsidy, CHC establishment
- KCC: KCC application, AHDF-KCC application
- MIDH: NHB account registration, GoC application preparation, GoC
  availability check, subsidy claim (12 steps), track application,
  grievance registration
- NFSM: multi-state application variants (MahaDBT, Meghalaya, Tamil Nadu,
  Rajasthan)
- AIF: LI Interest-Subvention Claim, LI CGTMSE Fee Claim

Workflow honesty rule: where a portal step is not publicly exposed
(e.g. post-login forms), the workflow explicitly labels this rather than
inventing fields. Where fresh applications are suspended (e.g. NHB GoC
availability check), the workflow records that state rather than presenting
an idealised path.

---

## 4. Routing Rules — Where Each Layer Applies

The layers are complementary, not overlapping. The routing rule is
deterministic and defensible:

1. **Structured layer first** — if the question is about eligibility or
   attribute-matching ("Am I eligible for X?"), it is answered from the
   structured layer using deterministic logic. RAG is not used.
2. **Workflow layer next** — if the question maps to a known
   `workflow_id` ("How do I apply for X? How do I do e-KYC on the PM-KISAN
   portal?"), it is served from the workflow layer as ordered steps. RAG
   is not used.
3. **RAG layer last** — everything else. Definitions, conditions,
   exceptions, cross-provision interactions, "what changed in 2018",
   "which document supersedes this rule", "what does *marginal farmer*
   mean in this scheme's context" — these require retrieval over prose
   with citation.

Some information can appear in *more than one layer*. A benefit amount
may be stored structurally for filtering, held as a workflow input for
procedural steps, and preserved in RAG as the authoritative prose
explaining the condition and its exceptions. The structured and workflow
layers are *decision/procedural* representations; the RAG layer remains
the *authoritative explanatory* representation and is always the source
of truth for citations.

---

## 5. Why Retrieval Belongs Only on the Prose Layer

This is the load-bearing design decision of the project.

Eligibility matching belongs in the structured layer because it is a
deterministic filtering problem over explicit attributes. Encoding it as
retrieval would be forcing RAG where a database query is correct.

Portal workflows belong in the workflow layer because they are ordered,
deterministic procedural knowledge with known inputs and conditions.
Retrieving approximate step sequences from prose is worse than serving
the actual steps.

RAG belongs on the prose layer because procedures, definitions,
conditions, exceptions and evolving operational rules require finding the
relevant passages from authoritative documents. Retrieval is the correct
tool here — and only here.

This three-layer separation prevents the RAG system from being used as a
substitute for deterministic logic while still giving it a meaningful,
non-trivial role where retrieval is actually necessary. Knowing where
retrieval does *not* belong is itself a core design decision.

---

## 6. Corpus Sources

**Primary source: Department of Agriculture & Farmers Welfare (DA&FW),
Ministry of Agriculture & Farmers Welfare, Government of India.** Includes
official scheme portals, operational guidelines, amendments, circulars,
FAQs and related official documents.

**Regulatory source for KCC: Reserve Bank of India (RBI).** KCC is
DA&FW-implemented but RBI-regulated. The four 2026 bank-type-specific
Directions, the 2017 Master Circular and the 2022 Circular are RBI
documents. This is a deliberate expansion of the source hierarchy
specific to KCC.

**Implementing-agency sources:**

- **NHB (National Horticulture Board)** for MIDH — operational guidelines,
  subsidy-claim SOP, grievance mechanism.
- **State agriculture departments and state DBT portals** for NFSM —
  MahaDBT (Maharashtra), Meghalaya Agriculture Department, Tamil Nadu,
  Rajasthan.
- **DBT Agriculture Mechanization portal** for SMAM.

**Supporting technical sources:**

- **NCCD (National Committee on Cold-chain Development)** engineering
  guidelines and basic datasheets, filed under MIDH but cross-referenced
  from AIF-funded cold-storage projects.

**FAQ treatment:** Scheme FAQs are treated as first-class corpus material
for this project. Unlike regulatory FAQs where the authority-versus-
interpretation distinction matters, scheme FAQs are frequently the
clearest procedural source and are cited alongside operational guidelines
without a separation.

**Third-party summaries, mirrors, blogs and consulting notes are not
authoritative and are excluded from the corpus.** Third-party pages may
be used only to *discover* an official source; the final corpus uses the
official version.

---

## 7. Corpus Principle

The goal is not to collect the largest possible number of PDFs.

The goal is to build a **small, authoritative and heterogeneous corpus**
whose documents contain enough procedural detail, exceptions, definitions
and cross-condition reasoning to make retrieval genuinely useful.

The seven schemes were selected for **knowledge diversity**, not scheme
count — income support, insurance, credit, mechanization, horticulture,
food security, and infrastructure financing each contribute a distinct
regulatory shape. AIF's financing-scheme shape is particularly important:
it is the only non-subsidy scheme in the corpus and broadens the
technical claims of the retrieval system beyond subsidy-shape questions.

The corpus is intentionally curated by the project owner from official
sources, not automated. Every included document has an inclusion reason
that the owner can defend.

---

## 8. Known Limitations of V1

The V1 corpus has explicit, honest limitations. Golden-set questions
must respect these — the system cannot answer questions the corpus does
not support, and it should not be tested on questions outside its
declared coverage.

- **MIDH workflow coverage is NHB-only.** MIDH is also implemented via
  NHM (National Horticulture Mission), HMNEH (Horticulture Mission for
  Northeast and Himalayan States), CDB (Coconut Development Board) and
  CIH (Central Institute of Horticulture). Workflows for those agencies
  are not captured in V1.
- **NFSM workflow coverage is limited to four states** (Maharashtra,
  Rajasthan, Meghalaya, Tamil Nadu) out of 28+ states. Other states are
  not captured. The prose corpus for NFSM is national.
- **AIF temporal chain is truncated** to Jan 2023 → Sept 2024 in V1. The
  2020 launch guidelines, 31-July-2020 clarification, 6-Aug-2020
  application-process circular and May 2022 Revised Guidelines were
  identified but not ingested in V1. AIF temporal questions before 2023
  are outside V1 coverage.
- **Priority-OCR list resolved (2026-09-09).** The V1 priority-OCR
  checklist has been processed:
  - **OCR'd via `ocrmypdf` (Tesseract, English + Hindi):** PMFBY
    YESTECH Manual (120p), SMAM 2016-17 Operational Guidelines,
    MIDH 2025 Operational Guideline (99p), MIDH
    FinalNHBOperationalGuideline (96p), AIF Sept 2024 Revised Scheme
    Guidelines. Each has a searchable-text sibling with `_OCR.pdf`
    suffix in the same folder; the original is preserved for audit
    trail. The Phase 2 loader must prefer the `_OCR.pdf` version
    when both exist.
  - **Removed rather than OCR'd:** the two PMFBY 2016 Notification
    files — see "PMFBY pre-2020 exclusion" below.
- **Non-priority scanned PDFs remain.** Approximately 25 non-priority
  scanned PDFs across the corpus (mostly PMFBY implementation
  letters and administrative circulars, a few PM_KISAN administrative
  notes, individual SMAM/MIDH/NFSM/AIF ancillary files) are still
  image-only and will surface as empty text to the Phase 2 loader.
  These are considered acceptable V1 losses; a second OCR pass on
  them is out of scope for V1 unless golden-set gaps prove them
  load-bearing.
- **PMFBY pre-2020 exclusion.** PMFBY temporal coverage in V1 begins
  at 2020 (Revamped Operational Guidelines). No pre-2020 PMFBY
  foundational documents are in the V1 corpus. The two 2016
  Notification files initially collected were opened and confirmed to
  be Uttar Pradesh-specific PMFBY implementation notifications
  (Hindi-only), not the national foundational PMFBY framework —
  outside V1 temporal/substantive scope and removed rather than
  OCR'd. Golden-set questions must not target information answerable
  only from these 2016 UP implementation notifications.
- **Filename normalisation pending** across schemes — several documents
  currently have cryptic, typo-bearing or non-descriptive filenames
  (`doc202679916301.pdf`, `FamilyDefination.pdf`, `Imp_Inst_LoCGoC.pdf`,
  `SchemeCircular.pdf`, `GUIEDELINES` typos, double `.pdf.pdf`
  extensions). A batch rename pass is planned before Phase 2 chunking.
- **Cross-scheme reference file (MIDH ↔ AIF cold-chain)** requires the
  exact MIDH/NCCD filepath to be resolved once all scheme folders are
  consolidated under one project root.

---

## 9. Version and Change Control

**V1.1 — locked after Phase 1 collection.**

Changes since V1.0:

- Per-scheme rationale updated with evidence from actual collected
  material (documents, workflow counts, temporal chains).
- Explicit scope decisions on adjacent schemes (PM-KMY, MISS, AHDF-KCC,
  NMEO-Oil Palm) recorded.
- Explicit V1 exclusions recorded (RWBCIS and general adjacent material).
- Workflow layer added as a first-class third layer alongside structured
  and RAG.
- Routing rules formalised as a three-step deterministic sequence
  (structured → workflow → RAG).
- Corpus source section expanded with the actual sources used, including
  RBI as regulatory source for KCC and NHB/state portals as implementing-
  agency sources.
- FAQ treatment made explicit (first-class corpus for scheme FAQs).
- Known Limitations section added, listing exactly what V1 does not
  cover so golden-set design respects real coverage.

Further scope revisions require a new version number (V1.2, V2.0 etc.)
and a corresponding entry in DECISIONS.md. Silent scope drift is not
permitted.
