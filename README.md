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
                                                      (720 pairs: 648 train / 72 test)
                                                              |
                                        [Stage 1: QLoRA SFT on Mistral 7B]
                                                  (Kaggle P100, ~67 min)
                                                              |
                                        [Stage 2: DPO on SFT Model]
                                                  (Kaggle P100 -- mode collapse)
                                                              |
                                                    [Evaluation]
                                              (ROUGE-L + GPT-4o judge)
```

## Results Summary

| Metric | Base Mistral 7B | After LoRA SFT | Change |
|--------|----------------|----------------|--------|
| ROUGE-L (mean) | 0.195 | 0.468 | +140% |
| ROUGE-L (median) | 0.181 | 0.440 | +143% |
| Avg word count | 196 | 132 | -33% |

**SFT achieves 2.4x higher ROUGE-L** against reference HR policy answers while producing more concise responses.

GPT-4o-mini pairwise judge: Base preferred 76.1% vs SFT 23.9% (71 prompts). This reflects a known LLM-as-judge bias toward longer, more verbose responses -- the base model gives generic textbook answers while SFT gives shorter, policy-specific answers that match ground truth better (as confirmed by ROUGE-L). See [EVALUATION_REPORT.md](EVALUATION_REPORT.md) for detailed analysis.

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
│   │   ├── candidates.json         # 9 candidates per query
│   │   ├── judgments.json          # GPT-4o judge verdicts
│   │   ├── train.json              # 648 training pairs
│   │   └── train.parquet
│   └── eval/                       # Evaluation artifacts
│       ├── test.json               # 72 held-out test pairs
│       ├── eval_results.json       # Base + SFT responses on test set
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

# Generate 9 candidate responses per query (20 queries)
python -m src.preference.collect

# Judge response pairs with GPT-4o
python -m src.preference.judge

# Format into HF Datasets (90/10 train/test split)
python -m src.preference.format
```

### Stage 2: QLoRA SFT Training (Kaggle GPU)

Upload `data/preference_data/train.json` and `data/eval/test.json` to Kaggle, then run `notebooks/kaggle_sft.ipynb`.

- Base model: Mistral 7B Instruct v0.2
- Quantization: 4-bit NF4 with double quantization
- LoRA: r=16, alpha=32, targets q/k/v/o projections
- Training: 3 epochs, batch=4, grad_accum=4, lr=2e-4, cosine schedule
- Time: ~67 minutes on Tesla P100 (123 steps)
- Output: ~27 MB LoRA adapter

### Stage 3: DPO Training (Kaggle GPU)

Run `notebooks/kaggle_dpo.ipynb` using the SFT adapter as starting point.

- DPO beta: 0.5 (increased from 0.1 to stay closer to SFT)
- Learning rate: 1e-7 (reduced from 5e-7)
- 1 epoch

**Note:** DPO training resulted in mode collapse. The DPO model is excluded from final evaluation.

### Stage 4: Evaluation

```bash
# Run on Kaggle GPU: generate base + SFT responses on test set
# notebooks/kaggle_eval.ipynb -> produces eval_results.json

# Local: compute ROUGE-L metrics
python -m src.eval.metrics --results data/eval/eval_results.json --references data/eval/test.json

# Local: GPT-4o pairwise preference judging
python -m src.eval.compare --results data/eval/eval_results.json
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

1. **SFT works** -- 2.4x ROUGE-L improvement, model learns company-specific HR policies
2. **DPO failed** -- mode collapse with both beta=0.1 and beta=0.5; excluded from evaluation
3. **LLM-as-judge has verbosity bias** -- GPT-4o prefers longer generic answers over shorter policy-specific ones
4. **ROUGE-L is the reliable metric** -- directly measures alignment with ground-truth reference answers
5. **QLoRA is efficient** -- full SFT training in ~67 minutes, adapter is only ~27 MB
