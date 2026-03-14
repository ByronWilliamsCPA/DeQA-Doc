def __getattr__(name):
    if name == "MPLUGOwl2LlamaForCausalLM":
        from .model import MPLUGOwl2LlamaForCausalLM

        return MPLUGOwl2LlamaForCausalLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
