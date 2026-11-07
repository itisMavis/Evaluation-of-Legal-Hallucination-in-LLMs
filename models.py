# Adapted from AraHalluEval (Alansari & Luqman, ArabicNLP 2025)
# https://github.com/aishaalansari57/AraHalluEval
# Modified for closed-book legal QA on the Saudi Personal Status Law.
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

login(token=os.environ.get("HF_TOKEN", ""))

API_PREFIXES = ("openai", "together", "claude", "deepseek")

# Final evaluated set: 2 Arabic-specialized, 3 general multilingual, 2 reasoning.
MODEL_MAP = {
    # ---- Local Hugging Face models ----
    "allam": "humain-ai/ALLaM-7B-Instruct-preview",   # Arabic-specialized
    "fanar": "QCRI/Fanar-1-9B-Instruct",              # Arabic-specialized
    "qwen":  "Qwen/Qwen2.5-7B-Instruct",              # general multilingual
    "llama": "meta-llama/Meta-Llama-3-8B-Instruct",   # general multilingual

    # ---- API models ----
    "deepseek:v4-pro":    "deepseek-v4-pro",          # general multilingual (DeepSeek API)
    "openai:gpt-5.6-sol": "gpt-5.6-sol",              # reasoning
    "openai:gpt-6-sol":   "gpt-6-sol",                # reasoning

    # Earlier-scope candidates, NOT in the final evaluated set:
    # "openai:gpt-4o": "gpt-4o",
    # "together:deepseek-r1": "deepseek-ai/DeepSeek-R1-0528",
}


def is_api_model(model_key: str) -> bool:
    return model_key.startswith(API_PREFIXES)


def load_model_and_tokenizer(model_key):
    """Load a local HF model; API models return (None, None, None)."""
    if is_api_model(model_key):
        return None, None, None

    model_name = MODEL_MAP[model_key]
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, return_token_type_ids=False
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    return model.to(device), tokenizer, device