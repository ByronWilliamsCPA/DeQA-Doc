"""Uncertainty quantification and pseudo-labeling pipeline for DeQA-Doc.

Provides tools to detect where SigLIP2-IQA predictions may be unreliable
using cross-model validation against public DeQA model weights, Mahalanobis
OOD detection, and multi-signal uncertainty fusion.
"""
