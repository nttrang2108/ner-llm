from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "vlps_2018_ner" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "vlps_2018_ner" / "processed"

ENAMEX_PATTERN = re.compile(r'<ENAMEX\s+TYPE="([^"]+)">(.*?)</ENAMEX>', re.IGNORECASE | re.DOTALL)
ENTITY_KEY_MAP = {
    "PERSON": "person",
    "ORGANIZATION": "organizations",
    "ORGANISATION": "organizations",
    "ORG": "organizations",
    "LOCATION": "address",
    "LOC": "address",
}


def clean_text(text: str) -> str:
    """Clean text by removing/normalizing unwanted characters.

    - Replace newlines with spaces
    - Remove quotes (both straight and curly quotes)
    - Normalize multiple spaces to single space
    - Strip leading/trailing whitespace
    """
    # Replace newlines with spaces
    text = text.replace('\n', ' ')

    # Remove various types of quotes
    text = text.replace('"', '')
    text = text.replace('"', '')
    text = text.replace('"', '')

    # Normalize multiple spaces to single space
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    return text.strip()


def strip_tags_and_collect_entities(text: str) -> Tuple[str, Dict[str, List[str]]]:
    """Remove ENAMEX tags while collecting entity mentions."""
    ground_truth = {"person": [], "organizations": [], "address": []}

    # Process tags from innermost to outermost
    # Each iteration removes one layer and collects entities without nested tags
    current_text = text
    while ENAMEX_PATTERN.search(current_text):
        # Find all matches in current iteration
        matches = list(ENAMEX_PATTERN.finditer(current_text))

        for match in matches:
            entity_type = match.group(1).strip().upper()
            value = match.group(2).strip()

            # Only collect if the value has NO tags at all (check for < character)
            # This ensures we only collect innermost entities with clean values
            if '<' not in value:
                key = ENTITY_KEY_MAP.get(entity_type)
                if key and value and value not in ground_truth[key]:
                    # Clean the entity value before adding
                    cleaned_value = clean_text(value)
                    if cleaned_value:
                        ground_truth[key].append(cleaned_value)

        # Remove one layer of tags (replace with content)
        current_text = ENAMEX_PATTERN.sub(lambda m: m.group(2), current_text)

    # Clean the final text
    return clean_text(current_text), ground_truth


def parse_file(file_path: Path, topic: str) -> Dict:
    """Parse a single .muc file into the target JSON structure."""
    raw_text = file_path.read_text(encoding="utf-8")

    # Split into lines and extract title (first line)
    lines = raw_text.split('\n', 1)
    title_raw = lines[0].strip() if lines else ""
    body_raw = lines[1].strip() if len(lines) > 1 else ""

    # Remove ENAMEX tags from title
    title_clean = title_raw
    while ENAMEX_PATTERN.search(title_clean):
        title_clean = ENAMEX_PATTERN.sub(lambda m: m.group(2).strip(), title_clean)

    # Clean the title text
    title_clean = clean_text(title_clean)

    # Process body text to get clean text and entities
    text_clean, ground_truth = strip_tags_and_collect_entities(body_raw)

    stem = file_path.stem
    try:
        record_id = int(stem)
    except ValueError:
        record_id = stem

    return {
        "id": record_id,
        "topic": topic,
        "title": title_clean,
        "text": text_clean,
        "ground_truth": ground_truth,
    }


def collect_split(split: str) -> List[Dict]:
    """Collect all records for a given split (train/dev/test)."""
    split_dir = RAW_DIR / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    records: List[Dict] = []
    for topic_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        # collect both .muc and .txt files
        muc_files = list(topic_dir.glob("*.muc"))
        txt_files = list(topic_dir.glob("*.txt"))
        for muc_file in sorted(muc_files + txt_files):
            records.append(parse_file(muc_file, topic_dir.name))
    return records


def write_json(split: str, records: List[Dict]) -> None:
    """Write records to processed/<split>.json."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / f"{split}.json"
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} records to {output_path}")


def main() -> None:
    for split in ("train", "dev", "test"):
        records = collect_split(split)
        write_json(split, records)


if __name__ == "__main__":
    main()

