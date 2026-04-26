import os
import torch

SRC_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.dirname(SRC_DIR)
CONFIGS_DIR = os.path.join(PROJECT_DIR, "configs")

# Don't change this parameter
MAX_TRAINING_TIME_SECONDS = 60 * 30
MAX_LENGTH = 512
INPUT_IDS = "input_ids"
ATTENTION_MASK = "attention_mask"
LABELS = "labels"

MEMORY_SNAPSHOT_DIR = os.path.join(PROJECT_DIR, "memory_snapshots")
SAVE_MEMORY_SNAPSHOT = False

RUN_MODE = "final"  # "smoke" or "final"
PARALLEL_MODE = "fsdp"  # "baseline", "deepspeed", "fsdp"
DEEPSPEED_CONFIG_PATH = os.path.join(CONFIGS_DIR, "ds_zero3.json")
FSDP_MODE = "full_shard auto_wrap"
USE_PRETOKENIZED_DATASET = False
USE_FLASH_ATTN = True
USE_BF16 = True
USE_FP16 = not USE_BF16
MODEL_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
NUM_PROCESSES = 2

# Don't change these parameters
DATA_DIR = os.path.join(PROJECT_DIR, "tokenized_data")
TOKENIZER_NAME = "ai-forever/rugpt3small_based_on_gpt2"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output_dir_hw2")
NUM_SHARDS = 32
VALIDATION_SIZE = 5000
SMOKE_TRAIN_SIZE = 128
SMOKE_EVAL_SIZE = 16

TRAINING_CONFIG = {
    "output_dir": os.path.join(OUTPUT_DIR, "gpt2-1b-russian"),
    "optim": "adamw_torch",
    "num_train_epochs": 1,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 16,
    "save_steps": 1000,
    "save_total_limit": 1,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "warmup_steps": 200,
    "logging_steps": 10,
    "eval_steps": 1000,
    "eval_strategy": "steps",
    "load_best_model_at_end": False,
    "metric_for_best_model": "eval_loss",
    "gradient_checkpointing": False,
    "gradient_accumulation_steps": 1,
    "dataloader_num_workers": 4,
    "torch_compile": False,
    "bf16": USE_BF16,
    "fp16": USE_FP16,
    "report_to": "wandb",
}

SMOKE_TRAINING_CONFIG = TRAINING_CONFIG.copy()
SMOKE_TRAINING_CONFIG.update(
    {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "save_steps": 100,
        "load_best_model_at_end": False,
        "eval_strategy": "no",
        "logging_steps": 10,
        "torch_compile": False,
        "warmup_steps": 0,
        "lr_scheduler_type": "constant",
        "ddp_find_unused_parameters": False,
        "save_strategy": "no",
    }
)

A100_CANDIDATE_1 = {
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 20,
    "learning_rate": 5e-5,
    "lr_scheduler_type": "linear",
    "warmup_steps": 200,
    "torch_compile": False,
    "optim": "adamw_torch",
}

A100_CANDIDATE_2 = {
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 20,
    "learning_rate": 5e-5,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 200,
    "torch_compile": False,
    "optim": "adamw_torch",
}

A100_CANDIDATE_3 = {
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 20,
    "learning_rate": 1e-4,
    "lr_scheduler_type": "linear",
    "warmup_steps": 200,
    "torch_compile": False,
    "optim": "adamw_torch",
}

A100_CANDIDATE_4 = {
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 20,
    "learning_rate": 4e-4,
    "lr_scheduler_type": "linear",
    "warmup_steps": 0,
    "torch_compile": False,
    "optim": "adamw_torch",
}

A100_CANDIDATE_5 = {
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 10,
    "learning_rate": 4e-4,
    "lr_scheduler_type": "linear",
    "warmup_steps": 0,
    "torch_compile": False,
    "optim": "adamw_torch",
}

PROMPTS_FOR_GENERATION = [
    "Москва — это",
    "Научное исследование показало, что",
    "В будущем искусственный интеллект будет",
]

FINAL_CANDIDATE = A100_CANDIDATE_5
FINAL_RUN_NAME = "a100-candidate-5-lr4e-4-warmup0-acc10"