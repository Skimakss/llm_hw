import os

import torch
import wandb
from transformers import Trainer, TrainingArguments

from callbacks import TimeoutCallback
from config import (
    DEEPSPEED_CONFIG_PATH,
    FINAL_CANDIDATE,
    FINAL_RUN_NAME,
    FSDP_MODE,
    MAX_TRAINING_TIME_SECONDS,
    OUTPUT_DIR,
    PARALLEL_MODE,
    PROMPTS_FOR_GENERATION,
    RUN_MODE,
    SMOKE_EVAL_SIZE,
    SMOKE_TRAIN_SIZE,
    SMOKE_TRAINING_CONFIG,
    TRAINING_CONFIG,
    VALIDATION_SIZE,
    USE_PRETOKENIZED_DATASET,
)

from data_utils import load_tokenized_dataset, prepare_dataset, prepare_tokenizer, split_dataset
from model_utils import create_model, setup_torch_backend
from runtime_utils import (
    build_final_training_config,
    get_seen_samples,
    get_steps_per_second,
    initialize_wandb,
    is_main_process,
    log_hardware_info,
    log_run_setup,
    save_generations,
    save_memory_snapshot,
    save_run_config,
    should_run_final_eval,
    should_save_memory_snapshot,
    start_memory_snapshot_recording,
    validate_hardware,
    validate_runtime_config,
    get_peak_memory_metrics_mb,
)

# Temporary compatibility workaround for FSDP + current torch/transformers stack
import torch.distributed.fsdp as torch_fsdp

if not hasattr(torch_fsdp, "register_fsdp_forward_method"):
    torch_fsdp.register_fsdp_forward_method = lambda *args, **kwargs: None

def train_model():
    validate_runtime_config()
    validate_hardware()
    log_hardware_info()

    if is_main_process():
        print(
            f"RANK={os.getenv('RANK', '0')}, "
            f"LOCAL_RANK={os.getenv('LOCAL_RANK', '0')}, "
            f"WORLD_SIZE={os.getenv('WORLD_SIZE', '1')}"
        )

    if RUN_MODE == "smoke":
        current_run_name = f"hw2-smoke-{PARALLEL_MODE}"
        current_training_config = SMOKE_TRAINING_CONFIG
    elif RUN_MODE == "final":
        current_run_name = f"{FINAL_RUN_NAME}-{PARALLEL_MODE}"
        current_training_config = build_final_training_config(
            TRAINING_CONFIG,
            FINAL_CANDIDATE,
        )
    else:
        raise ValueError(f"Unknown RUN_MODE: {RUN_MODE}")

    current_training_config = current_training_config.copy()
    current_training_config["output_dir"] = f"{OUTPUT_DIR}/{current_run_name}"

    if PARALLEL_MODE == "deepspeed":
        current_training_config["deepspeed"] = DEEPSPEED_CONFIG_PATH
    if PARALLEL_MODE == "fsdp":
        current_training_config["fsdp"] = FSDP_MODE

    log_run_setup(current_run_name, current_training_config)
    initialize_wandb(run_name=current_run_name, training_config=current_training_config)
    setup_torch_backend()
    start_memory_snapshot_recording()

    tokenizer = prepare_tokenizer()

    if not USE_PRETOKENIZED_DATASET and is_main_process():
        prepare_dataset(
            smoke_test=(RUN_MODE == "smoke"),
            smoke_size=SMOKE_TRAIN_SIZE if RUN_MODE == "smoke" else None,
        )

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    dataset = load_tokenized_dataset()

    train_dataset, eval_dataset = split_dataset(
        dataset,
        validation_size=SMOKE_EVAL_SIZE if RUN_MODE == "smoke" else VALIDATION_SIZE,
    )

    if RUN_MODE == "smoke":
        train_dataset = train_dataset.select(range(SMOKE_TRAIN_SIZE))
        eval_dataset = eval_dataset.select(range(SMOKE_EVAL_SIZE))

    model = create_model(tokenizer, is_smoke_test=(RUN_MODE == "smoke"))
    training_args = TrainingArguments(**current_training_config)

    if is_main_process():
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
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        train_result = trainer.train()
        train_result.metrics.update(get_peak_memory_metrics_mb())

        train_result.metrics["seen_samples"] = get_seen_samples(
            trainer.state.global_step,
            current_training_config,
        )
        train_result.metrics["steps_per_second_computed"] = get_steps_per_second(
            {
                "train_runtime": train_result.metrics.get("train_runtime"),
                "global_step": trainer.state.global_step,
            }
        )
        if wandb.run is not None and is_main_process():
            wandb.log({
                "peak_allocated_mb": train_result.metrics.get("peak_allocated_mb"),
                "peak_reserved_mb": train_result.metrics.get("peak_reserved_mb"),
                "seen_samples": train_result.metrics.get("seen_samples"),
                "train_steps_per_second_computed": train_result.metrics.get("steps_per_second_computed"),
            })

        if is_main_process():
            print("TRAIN METRICS:", train_result.metrics, flush=True)

        trainer.save_state()
        trainer.save_metrics("train", train_result.metrics)

        if RUN_MODE == "final":
            final_model_dir = os.path.join(training_args.output_dir, "final_model")
            trainer.save_model(final_model_dir)

            if is_main_process():
                tokenizer.save_pretrained(final_model_dir)

        if should_run_final_eval():
            print("Running final evaluation...")

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            eval_results = trainer.evaluate()
            eval_results.update(get_peak_memory_metrics_mb())

            if is_main_process():
                print(f"Final evaluation results: {eval_results}")
                trainer.save_metrics("eval", eval_results)

            if is_main_process():
                save_generations(
                    trainer.model,
                    tokenizer,
                    training_args.output_dir,
                    PROMPTS_FOR_GENERATION,
                )
    finally:
        if wandb.run is not None:
            wandb.finish()

        if should_save_memory_snapshot():
            save_memory_snapshot(f"{current_run_name}-{RUN_MODE}.pickle")


if __name__ == "__main__":
    train_model()