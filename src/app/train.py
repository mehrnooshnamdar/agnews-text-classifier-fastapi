import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from app.config import SETTINGS


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(
    output_dir: Path,
    epochs: float = 1.0,
    batch_size: int = 2,
    lr: float = 2e-5,
    max_length: int = 128,
    seed: int = 42,
    train_limit: int = 4000,
    eval_limit: int = 800,
) -> None:
    _seed_everything(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    SETTINGS.artifacts_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(SETTINGS.dataset_name)

    tokenizer = AutoTokenizer.from_pretrained(SETTINGS.model_checkpoint)

    def tokenize(batch):
        return tokenizer(
            batch[SETTINGS.text_col],
            truncation=True,
            max_length=max_length,
        )

    train_ds = ds["train"].shuffle(seed=seed).select(
        range(min(train_limit, len(ds["train"])))
    )
    test_ds = ds["test"].select(
        range(min(eval_limit, len(ds["test"])))
    )

    tokenized_train = train_ds.map(
        tokenize, batched=True, remove_columns=[SETTINGS.text_col]
    )
    tokenized_test = test_ds.map(
        tokenize, batched=True, remove_columns=[SETTINGS.text_col]
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        SETTINGS.model_checkpoint, num_labels=SETTINGS.num_labels
    )

    args = TrainingArguments(
        output_dir=str(SETTINGS.artifacts_dir / "hf_trainer_runs"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=data_collator,
    )

    trainer.train()

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    meta = {
        "settings": asdict(SETTINGS),
        "train_args": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "max_length": max_length,
            "seed": seed,
            "train_limit": train_limit,
            "eval_limit": eval_limit,
        },
    }
    (SETTINGS.artifacts_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    train(output_dir=SETTINGS.model_dir)
