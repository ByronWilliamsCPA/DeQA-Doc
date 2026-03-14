"""DIQA-5000 dataset expansion pipeline.

Versioned dataset generation: DIQA-5000_0 (base) → DIQA-5000_1 (Tier 1) → ...

Each tier adds new training samples via three streams:
    Stream 1: Controlled degradation (deterministic labels, weight=0.7)
    Stream 2: Synth-multiscript-v3 degradation replay (deterministic, weight=0.7)
    Stream 3: VLM pseudo-labels (consensus labeling, weight=0.5)
"""
