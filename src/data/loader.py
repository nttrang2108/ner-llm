"""Data loading utilities for Vietnamese NER dataset"""

import json
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "vlps_2018_ner" / "processed"


class NERDataLoader:
    """Load and manage NER dataset splits"""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = Path(data_dir)
        self._cache = {}

    def load_split(self, split: str) -> List[Dict]:
        """Load a specific split (train/dev/test)"""
        if split in self._cache:
            return self._cache[split]

        file_path = self.data_dir / f"{split}.json"
        if not file_path.exists():
            raise FileNotFoundError(f"Split file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self._cache[split] = data
        return data

    def load_all(self) -> Dict[str, List[Dict]]:
        """Load all splits"""
        return {
            'train': self.load_split('train'),
            'dev': self.load_split('dev'),
            'test': self.load_split('test')
        }


def load_processed_data(data_dir: Path = DATA_DIR) -> Dict[str, List[Dict]]:
    """Convenience function to load all data splits"""
    loader = NERDataLoader(data_dir)
    return loader.load_all()
