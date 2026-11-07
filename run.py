# Adapted from AraHalluEval (Alansari & Luqman, ArabicNLP 2025)
# https://github.com/aishaalansari57/AraHalluEval
# Modified for closed-book legal QA on the Saudi Personal Status Law.
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import argparse
from models import load_model_and_tokenizer, MODEL_MAP
from inference import get_response

# Task-specific files and prompts
datasets = {
    "law": "dataset_99.csv",   
}

cols = {
    "law": "question_ar",
}

# Closed-book: plain instruction, no article given, no leading hints.
# Kept identical across every model per the scope's methodology requirement.
prompts = {
    "law": "أجب على السؤال التالي بناءً على نظام الأحوال الشخصية السعودي الصادر عام 2022م، "
       "باللغة العربية:\n\nالسؤال: {text}\n\nالإجابة:",
}


def main(model_key, task):
    assert model_key in MODEL_MAP, f"{model_key} not found in MODEL_MAP."

    # Load model/tokenizer/device
    model, tokenizer, device = load_model_and_tokenizer(model_key)

    # Load data
    dataset_name = datasets[task]
    col_name = cols[task]
    df = pd.read_csv(dataset_name)
    prompt_template = prompts[task]

    # Setup output directory
    output_dir = Path(f"{task}_{model_key.replace(':', '_')}")
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / "results.csv"

    # Resume support: skip rows already answered (API calls cost money).
    if out_path.exists():
        done = pd.read_csv(out_path)
        responses = done["response"].tolist()
        start = len(responses)
        print(f"Resuming from row {start}")
    else:
        responses, start = [], 0

    for idx in tqdm(range(start, len(df)), initial=start, total=len(df), desc=model_key):
        text_input = df.loc[idx, col_name]
        prompt = prompt_template.format(text=text_input)
        try:
            response = get_response(prompt, model, tokenizer, device, model_key=model_key, task=task)
        except Exception as e:
            response = f"ERROR: {e}"
        responses.append(response)

        # Checkpoint every 10 questions so a crash doesn't lose progress/paid calls.
        if (idx + 1) % 10 == 0 or idx == len(df) - 1:
            partial = df.iloc[: len(responses)].copy()
            partial["response"] = responses
            partial.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n✅ Done! Saved output to: {out_path}")
    print(f"Next: copy the 'response' column into your Grading sheet for model={model_key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KSA law questions through an LLM for hallucination evaluation.")
    parser.add_argument("--model", type=str, required=True, help="Model key from MODEL_MAP (e.g., openai:gpt-4o)")
    parser.add_argument("--task", type=str, choices=["law"], default="law")
    args = parser.parse_args()

    main(args.model, args.task)
