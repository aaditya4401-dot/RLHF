# Project Brief: Self-Improving AI System with LoRA SFT + DPO Fine-Tuning

## Overview

Build a self-improving AI system that uses a RAG pipeline (powered by Claude API) to collect human preference data, then fine-tunes an open-source 7B model (Mistral 7B) using a two-stage approach: LoRA Supervised Fine-Tuning (SFT) followed by Direct Preference Optimization (DPO). All training runs on Kaggle's free GPU tier within a 30-hour weekly budget.

---

## Architecture

```
[User Query] → [RAG System (Claude API)] → [Response Generation]
                                                    ↓
                                          [Preference Collection]
                                                    ↓
                                          [Preference Dataset]
                                                    ↓
                              [Stage 1: LoRA SFT on Mistral 7B]
                                                    ↓
                              [Stage 2: DPO on LoRA-tuned Model]
                                                    ↓
                                          [Aligned Model]
```

---

## Components

### 1. RAG System (Local Machine)

- **Purpose:** Retrieve relevant context and generate responses using Claude API
- **Stack:** LangChain / LlamaIndex, Claude API, Vector DB (FAISS or ChromaDB)
- **Tasks:**
  - Document ingestion and chunking
  - Embedding generation and vector storage
  - Query → Retrieval → Claude API response generation
  - Serve as the "teacher" model for preference data

### 2. Preference Data Collection Pipeline (Local Machine)

- **Purpose:** Generate paired preference data (chosen vs rejected responses)
- **Method:**
  - For each query, generate multiple responses (varying temperature, prompts)
  - Use Claude API as a judge to rank responses OR collect human feedback
  - Format as preference pairs: `(prompt, chosen_response, rejected_response)`
- **Target:** 5,000–10,000 preference pairs minimum
- **Output Format:** Hugging Face Dataset compatible (JSON/Parquet)

### 3. Stage 1 — LoRA Supervised Fine-Tuning (Kaggle GPU)

- **Purpose:** Teach the base model your domain-specific task
- **Base Model:** Mistral 7B (or Mistral 7B Instruct)
- **Method:** LoRA fine-tuning on task-specific instruction data
- **Config:**
  - LoRA rank: r=16 or r=32
  - LoRA alpha: 32 or 64
  - Target modules: `q_proj, k_proj, v_proj, o_proj`
  - 4-bit quantization via bitsandbytes (QLoRA)
  - Batch size: 4 (gradient accumulation: 4)
  - Learning rate: 2e-4
  - Epochs: 3
- **Estimated Time:** 6–8 hours on Kaggle K80/P100
- **Output:** LoRA adapter weights (save checkpoint!)

### 4. Stage 2 — DPO Fine-Tuning (Kaggle GPU)

- **Purpose:** Align the LoRA-tuned model using human preference data
- **Method:** Direct Preference Optimization (simpler than PPO, no separate reward model needed)
- **Input:** Preference dataset from Step 2 + LoRA-tuned model from Step 3
- **Config:**
  - DPO beta: 0.1 (controls deviation from reference model)
  - LoRA rank: same as SFT (r=16 or r=32)
  - 4-bit quantization via bitsandbytes
  - Learning rate: 5e-7 (lower than SFT)
  - Epochs: 1–2
- **Library:** TRL (Hugging Face) — `DPOTrainer`
- **Estimated Time:** 6–8 hours on Kaggle K80/P100
- **Output:** DPO-aligned LoRA adapter weights

### 5. Evaluation Pipeline

- **Purpose:** Show measurable improvement at each stage
- **Three-way comparison:**
  1. Base Mistral 7B (baseline)
  2. After LoRA SFT (task performance improvement)
  3. After DPO (alignment/quality improvement)
- **Metrics:**
  - Automated: perplexity, ROUGE, BLEU, task-specific metrics
  - Preference win rate: use Claude API to judge outputs from all 3 versions
  - Side-by-side comparisons on held-out test set
- **Key Rule:** Test set must be held out from both training stages

---

## Compute Budget (Kaggle — 30 hrs/week GPU)

| Task | Estimated Hours |
|------|----------------|
| Data preparation & upload | 2–3 hrs |
| LoRA SFT training | 6–8 hrs |
| DPO training | 6–8 hrs |
| Evaluation & testing | 2–3 hrs |
| **Total** | **16–22 hrs** |
| **Buffer remaining** | **8–14 hrs** |

---

## Tech Stack

| Component | Tool/Library |
|-----------|-------------|
| RAG Framework | LangChain or LlamaIndex |
| Teacher Model | Claude API (Anthropic) |
| Vector Store | FAISS or ChromaDB |
| Base Model | Mistral 7B (Hugging Face) |
| Fine-tuning | TRL (Hugging Face) — SFTTrainer, DPOTrainer |
| Quantization | bitsandbytes (4-bit QLoRA) |
| LoRA | PEFT (Hugging Face) |
| Experiment Tracking | Weights & Biases (free tier) |
| Training Compute | Kaggle GPU (K80/P100) |
| Dataset Format | Hugging Face Datasets |

---

## File Structure

```
project/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                    # Raw documents for RAG
│   ├── preference_data/        # Preference pairs (chosen/rejected)
│   └── eval/                   # Held-out test set
├── src/
│   ├── rag/
│   │   ├── ingest.py           # Document ingestion & chunking
│   │   ├── retriever.py        # Vector store & retrieval
│   │   └── generate.py         # Claude API response generation
│   ├── preference/
│   │   ├── collect.py          # Generate preference pairs
│   │   ├── judge.py            # Claude API as preference judge
│   │   └── format.py           # Convert to HF Dataset format
│   ├── training/
│   │   ├── sft_lora.py         # Stage 1: LoRA SFT script
│   │   ├── dpo_train.py        # Stage 2: DPO training script
│   │   └── config.py           # Training hyperparameters
│   └── eval/
│       ├── benchmark.py        # Run all 3 model versions
│       ├── metrics.py          # Compute automated metrics
│       └── compare.py          # Side-by-side preference eval
├── notebooks/
│   ├── kaggle_sft.ipynb        # Kaggle notebook for SFT
│   └── kaggle_dpo.ipynb        # Kaggle notebook for DPO
└── models/
    ├── sft_adapter/            # LoRA weights after SFT
    └── dpo_adapter/            # LoRA weights after DPO
```

---

## Key Dependencies

```txt
# Local machine
langchain>=0.1.0
anthropic>=0.18.0
chromadb>=0.4.0
faiss-cpu>=1.7.0
datasets>=2.16.0

# Kaggle notebooks
transformers>=4.36.0
trl>=0.7.0
peft>=0.7.0
bitsandbytes>=0.41.0
accelerate>=0.25.0
wandb>=0.16.0
```

---

## Why LoRA SFT + DPO (Not Just One)

- **LoRA SFT alone:** Makes the model better at the task, but doesn't encode human preferences about output quality, tone, or helpfulness.
- **DPO alone:** Tries to align a model that may not understand the task well yet — weaker results.
- **LoRA SFT → DPO:** The standard two-stage approach used by labs. SFT builds task competence, DPO adds alignment. Together they outperform either alone.

---

## Why DPO Over PPO

- DPO is simpler — no separate reward model inference loop during training
- More stable training with limited compute
- Better suited for Kaggle's resource constraints
- TRL has mature DPO support that runs well on older GPUs
- PPO requires ~2x the memory (policy + reward model simultaneously)

---

## Fallback Plan

If Kaggle training fails (notebook instability, timeout, CUDA issues):

1. **You still have the preference dataset** — this is valuable on its own
2. **Show the RAG pipeline working** — demonstrates the data collection system
3. **Present the training scripts** — show the code is ready, just needs compute
4. **Use the SFT checkpoint** — if SFT succeeds but DPO fails, you still show improvement from SFT

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Kaggle notebook timeout | Save checkpoints every 500 steps; use `resume_from_checkpoint` |
| GPU OOM | Use 4-bit quantization + gradient checkpointing |
| CUDA version conflicts | Pin library versions in requirements.txt |
| Poor preference data quality | Use Claude API as judge with clear rubrics |
| DPO training instability | Start with low beta (0.1), monitor KL divergence |
| Lost progress | Save adapters to Hugging Face Hub after each stage |

---

## Success Criteria

1. RAG system retrieves relevant context and generates quality responses
2. Preference dataset of 5K+ pairs with clear quality signal
3. Measurable improvement from Base → SFT → DPO on held-out test set
4. All training completed within Kaggle's 30-hour GPU budget
5. Reproducible pipeline with clear documentation
