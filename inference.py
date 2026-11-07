# Adapted from AraHalluEval (Alansari & Luqman, ArabicNLP 2025)
# https://github.com/aishaalansari57/AraHalluEval
# Modified for closed-book legal QA on the Saudi Personal Status Law.
import re
import torch
from models import MODEL_MAP

# ---- API keys (hardcoded per preference; these files aren't shared) ----
OPENAI_API_KEY = ""       # <- paste your OpenAI key
TOGETHER_API_KEY = ""     # <- paste your Together key
ANTHROPIC_API_KEY = ""    # <- only needed for a claude: model
DEEPSEEK_API_KEY = ""     # <- paste your DeepSeek key

# ---- Decoding config (locked, identical across every model) ----
TEMPERATURE = 0.0          # greedy / deterministic -> reproducible
MAX_TOKENS = 512           # non-reasoning: full legal answer + citation (sized to our gold answers)
REASONING_MAX_TOKENS = 16000  # reasoning models need room for hidden chain-of-thought
REPETITION_PENALTY = 1.0   # start clean (no suppression of legit legal repeats like الثلث/الثلث, fractions).
                           # The chat template fixes most looping on its own. If the pilot still shows
                           # degenerate loops, climb GENTLY: 1.0 -> 1.1 -> 1.15. Use the lowest value that
                           # stops looping while fractions/ordered lists survive. Avoid 1.2+ for this domain.

# model keys whose outputs include reasoning (need bigger budget + no temp override)
REASONING_KEYS = ("deepseek-r1", "qwq", "gpt-5", ":o3", "o4-mini", "grok4_reasoning", "reasoning", "gpt-6")


def is_reasoning(model_key: str) -> bool:
    k = model_key.lower()
    return any(t in k for t in REASONING_KEYS)


def _strip_think(text: str) -> str:
    """Remove <think>...</think> chain-of-thought some reasoning models emit."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_after_marker(text: str) -> str:
    """Keep only the answer after the Arabic marker; harmless fallback for all paths."""
    for key in ["الإجابة:", "الجواب:", "Answer:", "Final Answer:"]:
        if key in text:
            return text.split(key)[-1].strip()
    return text.strip()


def get_response(prompt, model, tokenizer, device, model_key, task="law", debug=True):
    try:
        # ---------------- OpenAI ----------------
        if model_key.startswith("openai"):
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            model_name = MODEL_MAP[model_key]          # resolve from MODEL_MAP, not substring guessing

            kwargs = dict(model=model_name, messages=[{"role": "user", "content": prompt}])
            if is_reasoning(model_key):
                # reasoning models use max_completion_tokens and reject temperature
                kwargs["max_completion_tokens"] = REASONING_MAX_TOKENS
            else:
                kwargs["max_tokens"] = MAX_TOKENS
                kwargs["temperature"] = TEMPERATURE
            response = client.chat.completions.create(**kwargs)
            decoded = (response.choices[0].message.content or "").strip()

        # ---------------- DeepSeek (OpenAI-compatible API) ----------------
        elif model_key.startswith("deepseek"):
            from openai import OpenAI
            client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
            model_name = MODEL_MAP[model_key]          # "deepseek-v4-pro"
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=REASONING_MAX_TOKENS if is_reasoning(model_key) else MAX_TOKENS,
                temperature=TEMPERATURE,               # V4-pro is general, not reasoning
            )
            decoded = (response.choices[0].message.content or "").strip()
            if debug:
                print(f"[DeepSeek response]: {decoded[:300]}")

        # ---------------- Together AI ----------------
        elif model_key.startswith("together"):
            from together import Together
            client = Together(api_key=TOGETHER_API_KEY)
            model_name = MODEL_MAP[model_key]
            response = client.chat.completions.create(   # chat endpoint -> applies chat template server-side
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=REASONING_MAX_TOKENS if is_reasoning(model_key) else MAX_TOKENS,
                temperature=TEMPERATURE,
                repetition_penalty=REPETITION_PENALTY,
            )
            decoded = (response.choices[0].message.content or "").strip()
            if debug:
                print(f"[Together AI response]: {decoded[:300]}")

        # ---------------- Anthropic (optional) ----------------
        elif model_key.startswith("claude"):
            import anthropic
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            model_name = MODEL_MAP[model_key]
            response = client.messages.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            decoded = response.content[0].text.strip()

        # ---------------- Local Hugging Face (ALLaM, Qwen, Fanar, Llama) ----------------
        else:
            messages = [{"role": "user", "content": prompt}]
            try:
                # apply the model's OWN chat template -> formatted STRING (robust across versions)
                text = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
            except Exception:
                # fallback only if a model genuinely has no chat template
                text = prompt

            enc = tokenizer(text, return_tensors="pt").to(device)   # gives input_ids + attention_mask
            output_ids = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                do_sample=False,                 # greedy (expresses temperature 0, no warning)
                max_new_tokens=MAX_TOKENS,
                repetition_penalty=REPETITION_PENALTY,
                pad_token_id=(tokenizer.pad_token_id or tokenizer.eos_token_id),
            )
            # decode ONLY the newly generated tokens, not the prompt
            gen_ids = output_ids[0, enc["input_ids"].shape[-1]:]
            decoded = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # ---------------- Post-processing ----------------
        decoded = _strip_think(decoded)          # no-op unless a reasoning model emitted <think>
        summary = _extract_after_marker(decoded)

        if debug:
            print(f"[{model_key}]: {summary[:300]}")

        return summary

    except Exception as e:
        import traceback
        return f"ERROR: {repr(e)}\n{traceback.format_exc()}"