"""Training hyperparameters and configuration.

Central config for both SFT and DPO stages. Kaggle notebooks import
from here, or you can copy the values directly into the notebook cells.
"""

# ── Base Model ──────────────────────────────────────────────────────────
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# ── Quantization (QLoRA — 4-bit NF4) ───────────────────────────────────
LOAD_IN_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_COMPUTE_DTYPE = "float16"
BNB_4BIT_USE_DOUBLE_QUANT = True

# ── LoRA ────────────────────────────────────────────────────────────────
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
LORA_BIAS = "none"
LORA_TASK_TYPE = "CAUSAL_LM"

# ── SFT Training ───────────────────────────────────────────────────────
SFT_EPOCHS = 3
SFT_BATCH_SIZE = 4
SFT_GRADIENT_ACCUMULATION_STEPS = 4
SFT_LEARNING_RATE = 2e-4
SFT_MAX_SEQ_LENGTH = 1024
SFT_WARMUP_RATIO = 0.03
SFT_LR_SCHEDULER = "cosine"
SFT_WEIGHT_DECAY = 0.01
SFT_LOGGING_STEPS = 25
SFT_SAVE_STEPS = 500
SFT_FP16 = True
SFT_GRADIENT_CHECKPOINTING = True
SFT_OUTPUT_DIR = "./sft_output"
SFT_ADAPTER_DIR = "./models/sft_adapter"

# ── DPO Training ───────────────────────────────────────────────────────
DPO_BETA = 0.1
DPO_EPOCHS = 2
DPO_BATCH_SIZE = 4
DPO_GRADIENT_ACCUMULATION_STEPS = 4
DPO_LEARNING_RATE = 5e-7
DPO_MAX_LENGTH = 1024
DPO_MAX_PROMPT_LENGTH = 512
DPO_WARMUP_RATIO = 0.1
DPO_LR_SCHEDULER = "cosine"
DPO_WEIGHT_DECAY = 0.01
DPO_LOGGING_STEPS = 25
DPO_SAVE_STEPS = 500
DPO_FP16 = True
DPO_GRADIENT_CHECKPOINTING = True
DPO_OUTPUT_DIR = "./dpo_output"
DPO_ADAPTER_DIR = "./models/dpo_adapter"

# ── Weights & Biases ──────────────────────────────────────────────────
WANDB_PROJECT = "air-india-rlhf"
WANDB_RUN_NAME_SFT = "sft-lora-mistral7b"
WANDB_RUN_NAME_DPO = "dpo-lora-mistral7b"

# ── Hugging Face Hub (optional backup) ────────────────────────────────
HF_HUB_REPO_SFT = ""  # e.g. "your-username/air-india-sft-adapter"
HF_HUB_REPO_DPO = ""  # e.g. "your-username/air-india-dpo-adapter"
