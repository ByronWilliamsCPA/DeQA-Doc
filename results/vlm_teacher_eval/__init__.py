"""VLM teacher evaluation infrastructure for document IQA.

Provides modular components for evaluating frontier VLMs as pseudo-label
teachers for SigLIP2-IQA distillation. Supports Anthropic API and
OpenRouter as providers.

Modules:
    prompts: IQA rating prompt templates
    image_utils: Image encoding for API transmission
    response_parser: JSON response parsing and validation
    vlm_client: Provider-specific API clients
    correlation: SRCC/PLCC correlation metrics
    run_eval: Main evaluation script (local + Modal)
"""
