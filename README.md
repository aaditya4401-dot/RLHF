# Self-Improving HR Policy Q&A System

## LoRA SFT + DPO Fine-Tuning on Mistral 7B

A self-improving HR policy question-answering system that uses a RAG pipeline (OpenAI GPT-4o) to generate high-quality responses, collects preference data via automated judging, and fine-tunes Mistral 7B using QLoRA Supervised Fine-Tuning (SFT) followed by Direct Preference Optimization (DPO). All training runs on Kaggle's free GPU tier (Tesla P100, 16 GB VRAM).

## Architecture

```
[HR Question] --> [RAG System (GPT-4o + ChromaDB)] --> [Response Generation]
                                                              |
                                                    [Preference Collection]
                                                       (9 candidates/query)
                                                              |
                                                    [Automated Judging]
                                                       (GPT-4o as judge)
                                                              |
                                                    [Preference Dataset]
                                                     (1,780 train / 198 test)
                                                              |
                                        [Stage 1: QLoRA SFT on Mistral 7B]
                                                  (Kaggle P100, ~67 min)
                                                              |
                                        [Stage 2: DPO on SFT Model]
                                                  (Kaggle P100, ~45 min)
                                                              |
                                                    [Evaluation]
                                              (ROUGE-L + GPT-4o judge)
```

## Results Summary

| Metric | Base Mistral 7B | After LoRA SFT | After DPO |
|--------|----------------|----------------|-----------|
| ROUGE-L (mean) | 0.164 | **0.293** | 0.200 |
| ROUGE-L (median) | 0.157 | **0.249** | 0.187 |
| Avg word count | 202 | **131** | 178 |

**SFT achieves a 1.78x higher ROUGE-L** against reference HR policy answers while producing the most concise responses. **DPO** trades some ROUGE-L for preference quality and structure, with zero hallucination instances (vs. 11 for Base).

GPT-4o-mini pairwise judge win rates (n=198, 95% Wilson intervals):

| Comparison | Win rate | 95% CI | p (vs 50%) |
|---|---|---|---|
| Base over SFT | 72.7% | [66.1, 78.5] | 1.2e-10 |
| DPO over SFT | 62.6% | [55.7, 69.1] | 0.00047 |
| DPO over Base | 57.6% | [50.6, 64.3] | 0.039 |

Base beating SFT reflects a known LLM-as-judge bias toward longer, more verbose responses: the base model gives generic textbook answers while SFT gives shorter, policy-specific ones that match ground truth better (as confirmed by ROUGE-L). DPO closes that gap and reverses it. Note that **DPO over Base is only marginally significant** — its interval reaches to 50.6%, barely clear of chance — so it should be read as a weak result, not a decisive one.

> **Read these numbers as a pipeline demonstration, not a benchmark.** The evaluation set is **not prompt-disjoint** from training: the 90/10 split was applied per preference-pair rather than per prompt, so all 55 questions appear in both splits and every evaluated prompt was seen during training. ROUGE-L is the most affected — it rewards reproducing references the model trained on. Roughly 45% of reference answers are also refusals ("refer to your HR department"), so part of what SFT learned is appropriate declining rather than policy knowledge. Full analysis in [EVALUATION_REPORT.md](EVALUATION_REPORT.md), Section 9.

## Data Sources

| Source | Type | Size | Location |
|--------|------|------|----------|
| strova-ai/hr-policies-qa-dataset | Policy documents | 533 docs | `data/raw/hr_policies/` |
| xwjzds/extractive_qa_question_answering_hr | Q&A pairs | 5,983 pairs | `data/raw/hr_qa_pairs.json` |
| GitLab Employee Handbook | Benefits, expenses, onboarding, code of conduct | 4 sections | `data/raw/handbooks/gitlab_*.txt` |
| Valve Employee Handbook | Full employee handbook | 1 PDF | `data/raw/handbooks/valve_employee_handbook.pdf` |

## Project Structure

```
├── data/
│   ├── raw/
│   │   ├── hr_policies/            # 533 HR policy documents (.txt)
│   │   └── handbooks/              # GitLab sections + Valve PDF
│   ├── chroma_db/                  # ChromaDB vector store
│   ├── preference_data/            # Preference pairs + HF Dataset
│   │   ├── candidates.json         # 9 candidates per query (55 queries)
│   │   ├── judgments.json          # 1,978 GPT-4o judge verdicts
│   │   ├── train.json              # 1,780 training pairs
│   │   └── train.parquet
│   └── eval/                       # Evaluation artifacts
│       ├── test.json               # 198 test records (55 prompts, not disjoint from train)
│       ├── eval_results.json       # Base + SFT + DPO responses on test set
│       ├── metrics.csv             # ROUGE-L + word count stats
│       └── preference_winrates.json
├── src/
│   ├── rag/                        # RAG pipeline
│   │   ├── ingest.py               # Document ingestion & chunking
│   │   ├── retriever.py            # ChromaDB vector retrieval
│   │   └── generate.py             # GPT-4o response generation
│   ├── preference/                 # Preference data pipeline
│   │   ├── collect.py              # Generate candidate responses
│   │   ├── judge.py                # Automated preference judging
│   │   └── format.py               # Convert to HF Datasets format
│   ├── training/                   # Fine-tuning code
│   │   ├── config.py               # Centralized hyperparameters
│   │   ├── sft_lora.py             # LoRA SFT training script
│   │   └── dpo_train.py            # DPO training script
│   └── eval/                       # Evaluation pipeline
│       ├── benchmark.py            # Generate responses from all models
│       ├── metrics.py              # ROUGE-L + length statistics
│       └── compare.py              # GPT-4o pairwise win-rate judging
├── notebooks/
│   ├── kaggle_sft.ipynb            # Stage 1: QLoRA SFT (Kaggle GPU)
│   ├── kaggle_dpo.ipynb            # Stage 2: DPO (Kaggle GPU)
│   └── kaggle_eval.ipynb           # Evaluation inference (Kaggle GPU)
├── models/
│   ├── sft_adapter/                # LoRA adapter after SFT
│   └── dpo_adapter/                # LoRA adapter after DPO
├── results/
│   ├── sft_adapter/                # Final SFT LoRA weights (~27 MB)
│   └── sft_output/                 # Training checkpoints
├── project_brief.md                # Full architecture & design document
├── EVALUATION_REPORT.md            # Detailed evaluation analysis & findings
└── requirements.txt
```

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate    # Linux/Mac
# venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

## Pipeline Stages

### Stage 1: RAG + Preference Collection (Local)

```bash
# Build vector store from HR documents
python -m src.rag.ingest

# Generate 9 candidate responses per query (55 queries)
python -m src.preference.collect

# Judge all 36 pairs per query with GPT-4o, keeping confidence >= 4/5
python -m src.preference.judge

# Format into HF Datasets (90/10 train/test split)
python -m src.preference.format
```

### Stage 2: QLoRA SFT Training (Kaggle GPU)

Upload `data/preference_data/train.json` and `data/eval/test.json` to Kaggle, then run `notebooks/kaggle_sft.ipynb`. The same logic is available as a script for any GPU box:

```bash
python -m src.training.sft_lora \
    --dataset data/preference_data/train.json \
    --adapter-dir models/sft_adapter
```

- Base model: Mistral 7B Instruct v0.2
- Quantization: 4-bit NF4 with double quantization
- LoRA: r=16, alpha=32, targets q/k/v/o projections
- Training: 3 epochs, batch=4, grad_accum=4, lr=2e-4, cosine schedule
- Time: ~67 minutes on Tesla P100 (123 steps)
- Output: ~27 MB LoRA adapter

> **Adapter provenance:** the shipped adapters were trained on an earlier
> 648-pair version of the dataset (123 steps at an effective batch of 16
> confirms this). The preference set was later regenerated to 1,780 pairs and
> the models were *not* retrained. Rerunning training on the current
> `train.json` will not reproduce the exact adapters in `models/`.

### Stage 3: DPO Training (Kaggle GPU)

Run `notebooks/kaggle_dpo.ipynb` using the merged SFT model as the base, with a fresh LoRA adapter trained for DPO (not reusing the SFT adapter directly). Script equivalent:

```bash
python -m src.training.dpo_train \
    --dataset data/preference_data/train.json \
    --sft-adapter models/sft_adapter \
    --adapter-dir models/dpo_adapter
```

The SFT adapter is merged into the base weights first, then a fresh LoRA is added and `ref_model=None` lets TRL derive the reference policy by disabling that adapter — so DPO gets both policies from one set of 7B weights instead of two. That is what makes it fit in 16 GB.

- DPO beta: 0.3 (0.1 caused divergence, 0.5 was too conservative)
- Learning rate: 5e-6
- 1 epoch
- Time: ~45 minutes on Tesla P100

DPO is preferred by the GPT-4o-mini judge over both SFT (62.6%) and Base (57.6%), with zero hallucination instances. See Section 4 of [EVALUATION_REPORT.md](EVALUATION_REPORT.md) for the full analysis.

### Stage 4: Evaluation

```bash
# Run on Kaggle GPU: generate base + SFT responses on test set
# notebooks/kaggle_eval.ipynb -> produces eval_results.json

# Local: compute ROUGE-L metrics
python -m src.eval.metrics --results data/eval/eval_results.json --references data/eval/test.json

# Local: GPT-4o pairwise preference judging (reports 95% Wilson CIs + binomial test)
python -m src.eval.compare --results data/eval/eval_results.json

# Re-derive CIs and p-values from existing judgments — no API calls
python -m src.eval.compare --recompute
```

## Tech Stack

| Component | Tool/Library |
|-----------|-------------|
| RAG Framework | LangChain |
| Teacher Model | OpenAI API (GPT-4o) |
| Vector Store | ChromaDB |
| Embeddings | all-MiniLM-L6-v2 (local, via sentence-transformers) |
| Base Model | Mistral 7B Instruct v0.2 |
| Fine-tuning | TRL (SFTTrainer, DPOTrainer) |
| Quantization | bitsandbytes (4-bit QLoRA) |
| LoRA | PEFT |
| Training Compute | Kaggle GPU (Tesla P100, 16 GB) |
| Evaluation Judge | GPT-4o-mini |

## Key Findings

1. **The pipeline runs end-to-end on free compute** -- RAG preference generation, QLoRA SFT, DPO, and evaluation in ~2.5 GPU-hours on a single Tesla P100, with a 27 MB adapter training only 0.19% of the model's 7.26B parameters. This is the claim the project best supports.
2. **SFT shifts the model toward the reference style** -- shorter responses (131 vs 202 words), higher lexical diversity, and 1.78x ROUGE-L. Measured on seen prompts, so partly memorization (see limitations below).
3. **LLM-as-judge has verbosity bias** -- GPT-4o-mini prefers longer, generic Base answers over shorter, policy-specific SFT ones. This finding stands independently of the split problem and was the most interesting result of the project.
4. **DPO wins the cleanest comparison** -- preferred over SFT 62.6% of the time with zero verbose off-topic responses. Since SFT and DPO trained on identical data, neither holds a memorization advantage, making this the least contaminated number in the evaluation.
5. **The evaluation has real limits** -- the test set is not prompt-disjoint (0 of 55 prompts held out), the corpus is small (55 unique questions), and ~45% of reference answers are refusals. Section 9 of [EVALUATION_REPORT.md](EVALUATION_REPORT.md) documents all of these and what fixing them requires.
