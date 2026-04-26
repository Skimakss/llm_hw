import os
from datasets import load_dataset
from transformers import AutoTokenizer

from config import (
    ATTENTION_MASK,
    DATA_DIR,
    INPUT_IDS,
    LABELS,
    MAX_LENGTH,
    NUM_SHARDS,
    TOKENIZER_NAME,
    VALIDATION_SIZE,
)


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
    import os

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
    import os

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

    if os.getenv("RANK", "0") == "0":
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(eval_dataset)}")

    return train_dataset, eval_dataset