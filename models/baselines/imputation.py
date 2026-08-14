"""
Imputation prompt builders for both AR and DM models.

Issue #23: given a BGL record with K fields dropped, impute the missing
fields. AR gets the known fields as context and must regenerate the full
JSON (paying for known fields too). DM gets a partial JSON with mask tokens
at the missing field positions — _denoise fills only those.
"""

import json

from metrics.schema import FIELD_NAMES
from models.baselines.ar_model import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES

MASK_TOKENS_PER_FIELD = 15


def build_ar_imputation_messages(raw_line: str, source: str,
                                  known_fields: dict,
                                  missing_fields: list[str]) -> list[dict]:
    """Build chat messages for AR imputation.

    AR receives the raw line + the known fields and must output the full
    JSON (regenerating known fields too — that's the cost we're measuring).
    """
    system = SYSTEM_PROMPT.replace(
        "Extract structured fields from raw log lines.",
        "Extract structured fields from raw log lines. "
        "Some fields are already known (provided below); "
        "output the COMPLETE JSON including both known and inferred fields."
    )

    known_json = json.dumps({k: v for k, v in known_fields.items() if v is not None and v != ""},
                           indent=2, default=str)
    missing_str = ", ".join(missing_fields)

    user = f"Source: {source}\n"
    user += f'Input: "{raw_line}"\n'
    user += f"Known fields (do NOT change these):\n{known_json}\n"
    user += f"Fields to infer: {missing_str}\n"
    user += "Output the complete JSON record with all 12 fields:\n"
    user += "Output:"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_dm_imputation_prompt_text(raw_line: str, source: str,
                                    known_fields: dict,
                                    missing_fields: list[str]) -> str:
    """Build the instruction prompt text for DM imputation.

    The partial JSON with mask tokens is built separately by
    build_dm_imputation_token_ids (which needs the tokenizer).
    """
    missing_str = ", ".join(missing_fields)
    known_json = json.dumps({k: v for k, v in known_fields.items() if v is not None and v != ""},
                           indent=2, default=str)

    user = f"Source: {source}\n"
    user += f'Input: "{raw_line}"\n'
    user += f"Known fields:\n{known_json}\n"
    user += f"Fields to fill: {missing_str}\n"
    user += "Complete the JSON record (fill the masked positions):\n"
    user += "Output:"

    return SYSTEM_PROMPT + "\n\n" + user


def build_dm_imputation_token_ids(tokenizer, prompt_text: str,
                                  known_fields: dict,
                                  missing_fields: list[str],
                                  mask_token_id: int,
                                  max_length: int = 2048) -> tuple:
    """Build input_ids for DM imputation with selective masking.

    Tokenizes the prompt, then builds the partial JSON token-by-token,
    inserting mask_token_id at missing-field positions and real tokens
    at known-field positions.

    Returns (input_ids tensor, prompt_len, n_mask_tokens).
    """
    import torch

    prompt_ids = tokenizer(
        prompt_text, return_tensors="pt", truncation=True,
        max_length=max_length)["input_ids"]
    prompt_len = prompt_ids.shape[1]

    missing_set = set(missing_fields)

    json_ids = []
    json_ids.extend(tokenizer.encode("{", add_special_tokens=False))

    for i, fname in enumerate(FIELD_NAMES):
        field_text = f'"{fname}": '
        json_ids.extend(tokenizer.encode(field_text, add_special_tokens=False))

        if fname in missing_set:
            json_ids.extend([mask_token_id] * MASK_TOKENS_PER_FIELD)
        else:
            val = known_fields.get(fname, "")
            if isinstance(val, bool):
                val_text = "true" if val else "false"
            elif isinstance(val, (int, float)):
                val_text = str(val)
            else:
                val_text = f'"{val}"'
            json_ids.extend(tokenizer.encode(val_text, add_special_tokens=False))

        if i < len(FIELD_NAMES) - 1:
            json_ids.extend(tokenizer.encode(", ", add_special_tokens=False))

    json_ids.extend(tokenizer.encode("}", add_special_tokens=False))

    json_tensor = torch.tensor([json_ids], dtype=prompt_ids.dtype)
    input_ids = torch.cat([prompt_ids, json_tensor], dim=1)

    n_mask = sum(1 for tid in json_ids if tid == mask_token_id)
    return input_ids, prompt_len, n_mask
