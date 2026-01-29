import json
import os
from typing import Any, Dict, Iterable, List, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "vqa_and_retrieval")
OUT_PATH = os.path.join(BASE_DIR, "camerabench_vqa.json")

FIXED_PROBLEM_YES = "Which of the following options is more likely to have the answer 'yes'?"
FIXED_PROBLEM_NO = "Which of the following options is more likely to have the answer 'no'?"
FIXED_DATA_SOURCE = "camerabench"
FIXED_DATASET = "camerabench"


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects from a .jsonl file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
                else:
                    # Skip non-dict entries
                    continue
            except json.JSONDecodeError:
                # Skip bad lines
                continue


def iter_json(path: str) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects from a .json file that may be an array or a single object."""
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
    elif isinstance(data, dict):
        yield data


def collect_records(src_root: str) -> List[Tuple[Dict[str, Any], str]]:
    records: List[Tuple[Dict[str, Any], str]] = []
    for root, _dirs, files in os.walk(src_root):
        for name in files:
            lower = name.lower()
            path = os.path.join(root, name)
            source = os.path.splitext(name)[0]
            if lower.endswith(".jsonl"):
                for obj in iter_jsonl(path):
                    records.append((obj, source))
            elif lower.endswith(".json"):
                for obj in iter_json(path):
                    records.append((obj, source))
    return records


def convert_record(
    obj: Dict[str, Any],
    problem_id: Any,
    use_pos_text: bool,
    use_pos_video: bool,
    force_problem_yes: bool | None = None,
) -> Dict[str, Any]:
    """Convert a source object to the target schema with controllable text/video pairing.

    - use_pos_text: True => use pos_text for option A, False => use neg_text for option A.
    - use_pos_video: True => use pos_video for path, False => use neg_video.
    - force_problem_yes: If provided, force problem statement to 'yes' or 'no'. If None, infer:
        * Matching (Pos+Pos or Neg+Neg) => 'yes'
        * Mismatching (Pos+Neg or Neg+Pos) => 'no'
    """
    pos_text = (
        obj.get("pos_text")
        or obj.get("pos_question")
        or (obj.get("texts") and isinstance(obj.get("texts"), list) and obj["texts"][0])
        or obj.get("skill")
        or "Option A"
    )
    neg_text = (
        obj.get("neg_text")
        or obj.get("neg_question")
        or (obj.get("texts") and isinstance(obj.get("texts"), list) and len(obj["texts"]) > 1 and obj["texts"][1])
        or "Option B"
    )

    # Choose which text is A/B based on use_pos_text
    a_text = pos_text if use_pos_text else neg_text
    b_text = neg_text if use_pos_text else pos_text

    # Extract video name per flag, with fallbacks
    video_name = None
    if use_pos_video:
        video_name = (
            obj.get("pos_video")
            or (obj.get("images") and isinstance(obj.get("images"), list) and obj["images"][0])
        )
    else:
        video_name = (
            obj.get("neg_video")
            or (obj.get("images") and isinstance(obj.get("images"), list) and len(obj.get("images", [])) > 1 and obj["images"][1])
        )
    video_name = video_name or ""

    options = [f"A. {a_text}", f"B. {b_text}"]

    # Decide problem text (yes/no) following matching rule unless forced
    if force_problem_yes is None:
        is_match = (use_pos_text and use_pos_video) or ((not use_pos_text) and (not use_pos_video))
        problem_text = FIXED_PROBLEM_YES if is_match else FIXED_PROBLEM_NO
    else:
        is_match = force_problem_yes
        problem_text = FIXED_PROBLEM_YES if force_problem_yes else FIXED_PROBLEM_NO

    # Set solution: ensure some entries have B as the answer
    solution = "A" if is_match else "B"

    converted = {
        "problem_id": problem_id,
        "problem": problem_text,
        "data_type": "video",
        "problem_type": "multiple choice",
        "options": options,
        "process": "",
        "solution": f"<answer>{solution}</answer>",
        "path": f"video/{video_name}" if video_name else "",
        "data_source": FIXED_DATA_SOURCE,
        "dataset": FIXED_DATASET,
        "output": [],
    }
    return converted


def main() -> None:
    records = collect_records(SRC_DIR)
    total_entries = len(records)
    output: List[Dict[str, Any]] = []
    per_source_index: Dict[str, int] = {}

    # For each record, generate up to 4 combinations:
    # Pos+Pos (match => yes), Pos+Neg (mismatch => no), Neg+Pos (mismatch => no), Neg+Neg (match => yes)
    for obj, source in records:
        try:
            combos = [
                (True, True),   # Pos text + Pos video => yes
                (True, False),  # Pos text + Neg video => no
                (False, True),  # Neg text + Pos video => no
                (False, False), # Neg text + Neg video => yes
            ]
            for use_pos_text, use_pos_video in combos:
                idx = per_source_index.get(source, 0) + 1
                per_source_index[source] = idx
                pid = f"{source}-{idx}"
                converted = convert_record(
                    obj,
                    pid,
                    use_pos_text=use_pos_text,
                    use_pos_video=use_pos_video,
                )
                output.append(converted)
        except Exception:
            continue

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Total entries read: {total_entries}")
    print(f"Wrote {len(output)} items to {OUT_PATH}")


if __name__ == "__main__":
    main()