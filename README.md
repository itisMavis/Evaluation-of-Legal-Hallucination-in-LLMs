# Saudi Personal Status Law Hallucination Benchmark

A benchmark for measuring hallucination in large language models on the Saudi Personal
Status Law (Royal Decree M/73, 2022). It contains 99 Arabic questions drawn directly from
the statute and is used to evaluate seven LLMs on how accurately they answer closed-book
legal questions.

This repository is the data and code release for the paper *"Legal Hallucination in LLMs
on Saudi Personal Status Law: An Arabic Statutory Benchmark"* ([venue / link]).

## What it measures

Each question is answered closed-book (no article text is given to the model), and each
answer is graded against the official statute on three axes: substantive accuracy (is the
rule correct), citation groundedness (are the cited articles real), and resistance to false
premises (does the model reject a planted falsehood).

## Dataset

`dataset_99.csv` holds 99 questions, balanced across three types with 33 each:

- **Direct** recall of a single rule
- **Scenario** questions that apply a rule in a short fact pattern
- **Fabrication probes** that plant a false premise, where the correct response rejects it

The items span 33 distinct articles. Columns: `id`, `topic`, `source_article`,
`question_ar`, `question_type`, `prob_type`, `gold_answer_ar`, `source_text_ar`,
`verifiable_solo`, `notes`. Every gold answer is stated plainly in a single cited article
and was checked against the official Arabic text published by the Bureau of Experts at the
Council of Ministers.

## Repository contents

| File | Description |
|---|---|
| `dataset_99.csv` | The 99-question benchmark with gold answers |
| `responses_graded.xlsx` | Model responses with verdicts and the 11 indicator flags |
| `Annotation_Codebook.docx` | Annotation codebook: verdict definitions, indicators, rules, examples |
| `run.py` | Runs the questions through a model |
| `inference.py` | Per-provider inference (local Hugging Face and API models) |
| `models.py` | Model registry (`MODEL_MAP`) |

## Models evaluated

- **Arabic-specialized:** ALLaM, Fanar
- **General multilingual:** Qwen2.5, Llama 3, DeepSeek-V4-pro
- **Reasoning:** GPT-5.6-sol, GPT-6-sol

## Running the pipeline

```bash
pip install -r requirements.txt
python run.py --model allam --task law
```

Outputs are written to `law_<model>/results.csv`. All models receive the same Arabic prompt
(closed-book, deterministic decoding, 512-token limit). Add your API keys in `inference.py`
and set your Hugging Face token via the `HF_TOKEN` environment variable.


## Acknowledgements

The evaluation pipeline (`run.py`, `inference.py`, `models.py`) is adapted from the
AraHalluEval codebase by Alansari and Luqman
(https://github.com/aishaalansari57/AraHalluEval), modified for closed-book Arabic legal QA
on the Saudi Personal Status Law. We thank the authors for releasing their code.


