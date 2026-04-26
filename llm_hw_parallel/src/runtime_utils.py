import json
import os

import torch
import wandb

from config import (
    DEEPSPEED_CONFIG_PATH,
    FSDP_MODE,
    MEMORY_SNAPSHOT_DIR,
    NUM_PROCESSES,
    PARALLEL_MODE,
    PROMPTS_FOR_GENERATION,
    RUN_MODE,
    SAVE_MEMORY_SNAPSHOT,
    USE_BF16,
    USE_FP16,
)


def is_main_process():
    return os.getenv("RANK", "0") == "0"


def get_global_batch_size(training_config):
    return (
        training_config["per_device_train_batch_size"]
        * training_config["gradient_accumulation_steps"]
        * NUM_PROCESSES
    )


def get_seen_samples(global_step, training_config):
    return global_step * get_global_batch_size(training_config)


def get_peak_memory_metrics_mb():
    if not torch.cuda.is_available():
        return {}

    return {
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / (1024 ** 2),
    }


def get_steps_per_second(train_metrics):
    train_runtime = train_metrics.get("train_runtime")
    global_step = train_metrics.get("global_step")
    if not train_runtime or not global_step:
        return None
    return global_step / train_runtime


def validate_runtime_config():
    if PARALLEL_MODE not in {"baseline", "deepspeed", "fsdp"}:
        raise ValueError(f"Unknown PARALLEL_MODE: {PARALLEL_MODE}")
    if USE_BF16 and USE_FP16:
        raise ValueError("USE_BF16 and USE_FP16 cannot both be True")
    if PARALLEL_MODE == "deepspeed" and not DEEPSPEED_CONFIG_PATH:
        raise ValueError("DEEPSPEED_CONFIG_PATH is not set")
    if PARALLEL_MODE == "fsdp" and not FSDP_MODE:
        raise ValueError("FSDP_MODE is not set")
    if not USE_BF16 and not USE_FP16:
        raise ValueError("At least one of USE_BF16 or USE_FP16 must be enabled")


def validate_hardware():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")


def log_hardware_info():
    if is_main_process() and torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


def log_run_setup(current_run_name, training_config):
    if is_main_process():
        print(f"run_name={current_run_name}")
        print(f"run_mode={RUN_MODE}")
        print(f"parallel_mode={PARALLEL_MODE}")
        print(f"output_dir={training_config['output_dir']}")
        print(f"global_batch_size={get_global_batch_size(training_config)}")


def should_run_final_eval():
    return RUN_MODE == "final"


def should_save_memory_snapshot():
    return SAVE_MEMORY_SNAPSHOT and is_main_process()


def get_accelerate_launch_command():
    return f"accelerate launch --num_processes {NUM_PROCESSES} parallel_train.py"


def start_memory_snapshot_recording():
    if should_save_memory_snapshot():
        torch.cuda.memory._record_memory_history()


def save_memory_snapshot(snapshot_name):
    if should_save_memory_snapshot():
        os.makedirs(MEMORY_SNAPSHOT_DIR, exist_ok=True)
        snapshot_path = os.path.join(MEMORY_SNAPSHOT_DIR, snapshot_name)
        torch.cuda.memory._dump_snapshot(snapshot_path)
        print(f"Memory snapshot saved to: {snapshot_path}")


def initialize_wandb(run_name="run", training_config=None):
    if not is_main_process():
        return
    
    wandb.login()
    wandb.init(
        project="llm-mini-pretrain",
        name=run_name,
        config=training_config,
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


def build_final_training_config(training_config, candidate_config):
    final_config = training_config.copy()
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
        "parallel_mode": PARALLEL_MODE,
        "num_processes": NUM_PROCESSES,
        "global_batch_size": get_global_batch_size(training_config),
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)