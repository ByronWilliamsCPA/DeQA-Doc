"""Uncertainty quantification and pseudo-labeling pipeline for DeQA-Doc.

Provides tools to detect where SigLIP2-IQA predictions may be unreliable
using cross-model validation against public DeQA model weights, Mahalanobis
OOD detection, and multi-signal uncertainty fusion.

Metadata schema types and I/O are available via direct submodule imports::

    from src.uncertainty.metadata_schema import ImageMetadataRecord, LabelSource
    from src.uncertainty.metadata_io import read_master_jsonl, write_master_jsonl
"""
