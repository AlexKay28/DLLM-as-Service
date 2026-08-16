"""Monkey-patch for vLLM + transformers 5.x compatibility.
Import this BEFORE importing vllm.
"""
import transformers.models.auto.configuration_auto as hf_config

_original_register = hf_config.AutoConfig.register

def safe_register(model_type, config, exist_ok=True):
    try:
        _original_register(model_type, config, exist_ok=exist_ok)
    except ValueError as e:
        if 'already used by a Transformers config' in str(e):
            pass
        else:
            raise e

hf_config.AutoConfig.register = safe_register
