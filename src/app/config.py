from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_checkpoint: str = "distilbert-base-uncased"
    dataset_name: str = "ag_news"
    text_col: str = "text"
    label_col: str = "label"
    num_labels: int = 4

    artifacts_dir: Path = Path("artifacts")
    model_dir: Path = Path("models/agnews-distilbert")


SETTINGS = Settings()
