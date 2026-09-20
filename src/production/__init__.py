"""Production pipeline — hybrid → rerank → generate against the
`agri_schemes_prod` collection.

Deliberately separate from the ablation pipeline dispatch (which lives
in `eval/run_eval.py` under `--pipeline_mode`). Ablation modes stay
untouched and reproducible; production is one config-driven callable
that composes the two best-measured techniques (hybrid + rerank) plus
grounded generation.
"""
