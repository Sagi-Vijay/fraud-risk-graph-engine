from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    root_dir: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = root_dir / "data"
    artifacts_dir: Path = root_dir / "artifacts"

    def create_dirs(self):
        (self.data_dir / "raw").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "processed").mkdir(parents=True, exist_ok=True)
        (self.artifacts_dir / "models").mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    cfg = Config()
    cfg.create_dirs()
    print("Project directories created")
