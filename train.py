import json
import os
import time

import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Qwen3Config,
    Qwen3ForCausalLM,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# Don't change this parameter
MAX_TRAINING_TIME_SECONDS = 60 * 30
MAX_LENGTH = 512
INPUT_IDS = "input_ids"
ATTENTION_MASK = "attention_mask"
LABELS = "labels"

RUN_MODE = "final"  # "smoke" or "final"

# Don't change these parameters
DATA_DIR = "./tokenized_data"
TOKENIZER_NAME = "ai-forever/rugpt3small_based_on_gpt2"
OUTPUT_DIR = "./output_dir"
NUM_SHARDS = 32
VALIDATION_SIZE = 5000

TRAINING_CONFIG = {
    "output_dir": f"{OUTPUT_DIR}/gpt2-1b-russian",
    "optim": "adamw_torch",
    "num_train_epochs": 1,
    "per_device_train_batch_size": 4,
    "save_steps": 1000,
    "save_total_limit": 1,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "warmup_steps": 200,
    "logging_steps": 10,
    "eval_steps": 1000,
    "eval_strategy": "steps",
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "gradient_checkpointing": False,
    "gradient_accumulation_steps": 1,
    "dataloader_num_workers": 4,
    "torch_compile": False,
    "bf16": True,
    "report_to": "wandb",
}

SMOKE_TRAINING_CONFIG = TRAINING_CONFIG.copy()
SMOKE_TRAINING_CONFIG.update(
    {
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "save_steps": 100,
        "load_best_model_at_end": False,
        "eval_strategy": "no",
        "logging_steps": 10,
        "torch_compile": False,
        "warmup_steps": 0,
        "lr_scheduler_type": "constant",
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


PROMPTS_FOR_GENERATION = [
    "Москва — это",
    "Научное исследование показало, что",
    "В будущем искусственный интеллект будет",
]

FINAL_CANDIDATE = A100_CANDIDATE_4
FINAL_RUN_NAME = "a100-candidate-4-lr4e-4-warmup0"


class TimeoutCallback(TrainerCallback):
    """Stop training after a specified timeout."""

    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed > self.timeout_seconds:
                control.should_training_stop = True
                print(f"Training stopped after {elapsed:.2f} seconds")
        return control


def prepare_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_function(examples, tokenizer):
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    tokenized[LABELS] = tokenized[INPUT_IDS].copy()
    return tokenized


def save_as_parquets(ds, output_dir=DATA_DIR, num_shards=NUM_SHARDS):
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_shards):
        shard = ds.shard(num_shards=num_shards, index=i)
        shard.to_parquet(f"{output_dir}/{i:05d}.parquet")


def prepare_dataset(smoke_test=False, smoke_size=None):
    dataset = load_dataset("wikimedia/wikipedia", "20231101.ru", split="train")

    if smoke_test:
        dataset = dataset.select(range(smoke_size))

    tokenizer = prepare_tokenizer()

    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        num_proc=8,
    )
    tokenized_dataset = tokenized_dataset.select_columns(
        [INPUT_IDS, LABELS, ATTENTION_MASK]
    )

    save_as_parquets(tokenized_dataset)


def load_tokenized_dataset(data_dir=DATA_DIR):
    parquet_files = sorted(
        [
            os.path.join(data_dir, name)
            for name in os.listdir(data_dir)
            if name.endswith(".parquet")
        ]
    )
    dataset = load_dataset("parquet", data_files=parquet_files)
    return dataset["train"]


def split_dataset(dataset, validation_size=VALIDATION_SIZE):
    dataset_size = len(dataset)

    eval_dataset = dataset.select(range(validation_size))
    train_dataset = dataset.select(range(validation_size, dataset_size))

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")

    return train_dataset, eval_dataset


def setup_torch_backend():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def create_model(tokenizer, is_smoke_test=False):
    model_config = {
        "hidden_size": 2048,
        "num_hidden_layers": 12,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "intermediate_size": 8192,
        "head_dim": 128,
        "hidden_act": "silu",
        "initializer_range": 0.02,
        "scale_attn_weights": True,
        "use_cache": True,
    }

    config = Qwen3Config(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **model_config,
    )

    if not is_smoke_test:
        model = Qwen3ForCausalLM._from_config(
            config,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
    else:
        model = Qwen3ForCausalLM._from_config(
            config,
            torch_dtype=torch.bfloat16,
        )

    print(f"Model pad token id: {model.config.pad_token_id}")

    with torch.no_grad():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total params: {total_params:,}")

    return model


def initialize_wandb(run_name="run"):
    wandb.login()
    wandb.init(
        project="llm-mini-pretrain",
        name=run_name,
    )


def save_generations(model, tokenizer, output_dir, prompts):
    generations_path = os.path.join(output_dir, "generations.txt")
    model.eval()

    with open(generations_path, "w", encoding="utf-8") as f:
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

            f.write(f"PROMPT: {prompt}\n")
            f.write(f"GENERATION: {generated_text}\n")
            f.write("-" * 80 + "\n")


def build_final_training_config(candidate_config):
    final_config = TRAINING_CONFIG.copy()
    final_config.update(candidate_config)
    return final_config


def save_run_config(output_dir, training_config, run_name, prompts):
    os.makedirs(output_dir, exist_ok=True)

    config_path = os.path.join(output_dir, "run_config.json")
    payload = {
        "run_name": run_name,
        "training_config": training_config,
        "prompts_for_generation": prompts,
        "run_mode": RUN_MODE,
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def train_model():
    if RUN_MODE == "smoke":
        current_run_name = "smoke-run"
        current_training_config = SMOKE_TRAINING_CONFIG
    elif RUN_MODE == "final":
        current_run_name = FINAL_RUN_NAME
        current_training_config = build_final_training_config(FINAL_CANDIDATE)
    else:
        raise ValueError(f"Unknown RUN_MODE: {RUN_MODE}")

    initialize_wandb(run_name=current_run_name)
    setup_torch_backend()

    tokenizer = prepare_tokenizer()
    dataset = load_tokenized_dataset()
    train_dataset, eval_dataset = split_dataset(dataset)
    model = create_model(tokenizer, is_smoke_test=(RUN_MODE == "smoke"))

    training_args = TrainingArguments(**current_training_config)

    save_run_config(
        training_args.output_dir,
        current_training_config,
        current_run_name,
        PROMPTS_FOR_GENERATION,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[TimeoutCallback(timeout_seconds=MAX_TRAINING_TIME_SECONDS)],
    )

    try:
        train_result = trainer.train()
        trainer.save_state()
        trainer.save_metrics("train", train_result.metrics)

        final_model_dir = os.path.join(training_args.output_dir, "final_model")
        trainer.save_model(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)

        if RUN_MODE == "final":
            print("Running final evaluation...")
            eval_results = trainer.evaluate()
            print(f"Final evaluation results: {eval_results}")
            trainer.save_metrics("eval", eval_results)
            save_generations(
                model,
                tokenizer,
                training_args.output_dir,
                PROMPTS_FOR_GENERATION,
            )
    finally:
        wandb.finish()


if __name__ == "__main__":
    train_model()