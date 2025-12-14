# AG News Text Classifier (DistilBERT) + FastAPI

End-to-end NLP project:
- Train a transformer classifier on **AG News** (4 classes)
- Evaluate with classification report + confusion matrix
- Serve predictions via **FastAPI** (`/predict`)

## Tech
- PyTorch, HuggingFace Transformers + Datasets
- FastAPI (inference API)
- Ruff + Black + Pytest + GitHub Actions (CI)

## Results (on 3,000 test samples)
**Accuracy:** 0.8980
    precision    recall  f1-score   support
    World 0.8993 0.8849 0.8920 747
    Sports 0.9432 0.9858 0.9640 775
    Business 0.8530 0.8348 0.8438 702
    Sci/Tech 0.8893 0.8802 0.8847 776

    accuracy 0.8980 3000
    macro avg 0.8962 0.8964 0.8961 3000
    weighted avg 0.8972 0.8980 0.8974 3000
