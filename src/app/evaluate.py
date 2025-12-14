from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

from app.config import SETTINGS


LABELS = ["World", "Sports", "Business", "Sci/Tech"]


@torch.inference_mode()
def evaluate(model_dir: Path, max_length: int = 256, limit: int = 3000) -> None:
    ds = load_dataset(SETTINGS.dataset_name, split="test").select(range(limit))

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    preds = []
    y_true = []

    for item in ds:
        text = item[SETTINGS.text_col]
        y_true.append(item[SETTINGS.label_col])

        inputs = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
        out = model(**inputs)
        pred = int(torch.argmax(out.logits, dim=-1).cpu().item())
        preds.append(pred)

    y_true = np.array(y_true)
    preds = np.array(preds)

    print("\nClassification report:")
    print(classification_report(y_true, preds, target_names=LABELS, digits=4))

    cm = confusion_matrix(y_true, preds)
    print("\nConfusion matrix:")
    print(cm)


if __name__ == "__main__":
    evaluate(model_dir=SETTINGS.model_dir)
