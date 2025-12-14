from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.config import SETTINGS

LABELS = ["World", "Sports", "Business", "Sci/Tech"]


class TextClassifier:
    def __init__(self, model_dir: Path):
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.eval()

    @torch.inference_mode()
    def predict(self, text: str, top_k: int = 2) -> Dict:
        inputs = self.tokenizer(text, truncation=True, max_length=256, return_tensors="pt")
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)

        top_k = max(1, min(top_k, probs.numel()))
        vals, idxs = torch.topk(probs, k=top_k)

        results: List[Tuple[str, float]] = []
        for i, v in zip(idxs.tolist(), vals.tolist()):
            results.append((LABELS[i], float(v)))

        pred_idx = int(torch.argmax(probs).item())
        retur


cat > src/app/inference.py <<'EOF'
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.config import SETTINGS

LABELS = ["World", "Sports", "Business", "Sci/Tech"]


class TextClassifier:
    def __init__(self, model_dir: Path):
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.eval()

    @torch.inference_mode()
    def predict(self, text: str, top_k: int = 2) -> Dict:
        inputs = self.tokenizer(text, truncation=True, max_length=256, return_tensors="pt")
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)

        top_k = max(1, min(top_k, probs.numel()))
        vals, idxs = torch.topk(probs, k=top_k)

        results: List[Tuple[str, float]] = []
        for i, v in zip(idxs.tolist(), vals.tolist()):
            results.append((LABELS[i], float(v)))

        pred_idx = int(torch.argmax(probs).item())
        return {
            "label": LABELS[pred_idx],
            "confidence": float(probs[pred_idx].item()),
            "top_k": [{"label": lab, "prob": p} for lab, p in results],
        }
