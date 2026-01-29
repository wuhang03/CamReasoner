import os
import json
import argparse
import re

# 统计当前目录下 camerabench_sft 中的 entry 总数（支持 .json 和 .jsonl）
def count_entries_in_camerabench_sft(base_dir: str = None) -> int:
    if base_dir is None:
        base_dir = os.path.dirname(__file__)
    target_dir = os.path.join(base_dir, "camerabench_sft")
    if not os.path.isdir(target_dir):
        print(f"Directory not found: {target_dir}")
        return 0

    def iter_entries_in_file(path: str):
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            yield obj
                    except Exception:
                        continue
        else:
            with open(path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except Exception:
                    return
            if isinstance(data, list):
                for e in data:
                    if isinstance(e, dict):
                        yield e
            elif isinstance(data, dict):
                yielded = False
                for key in ("data", "entries", "items", "results"):
                    val = data.get(key)
                    if isinstance(val, list):
                        for e in val:
                            if isinstance(e, dict):
                                yield e
                        yielded = True
                        break
                if not yielded:
                    yield data

    total = 0
    for fname in os.listdir(target_dir):
        if not (fname.endswith(".json") or fname.endswith(".jsonl")):
            continue
        fpath = os.path.join(target_dir, fname)
        for _ in iter_entries_in_file(fpath):
            total += 1
    return total

def load_json_by_name(filename: str, base_dir: str = None):
    """Load a JSON file by name from common locations.
    If filename is an absolute path, read directly.
    Otherwise, search relative to this module and known workspace paths.
    """
    if not filename:
        raise FileNotFoundError("No filename provided")
    if base_dir is None:
        base_dir = os.path.dirname(__file__)
    if os.path.isabs(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    candidates = [
        os.path.join(base_dir, filename),
        os.path.join(base_dir, "camerabench_sft", filename),
        os.path.join(base_dir, "..", "..", "onethinker", "LLaMA-Factory", "data", filename),
        os.path.join(base_dir, "..", "..", "onethinker", "Data", "train", "filtered_reasoning", filename),
    ]
    for path in map(os.path.abspath, candidates):
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"{filename} not found in known locations")

def load_camerabench_sft_json(base_dir: str = None, filename: str = "camerabench_sft.json"):
    return load_json_by_name(filename, base_dir)

def _normalize_pattern(filename: str) -> str:
    # remove extension and normalize separators to spaces
    base = os.path.splitext(filename)[0]
    base = base.replace("_", " ").replace("-", " ")
    return base.lower()

def _extract_text_from_entry(entry: dict) -> str:
    # Try messages array (ChatML like)
    msgs = entry.get("messages") or entry.get("message")
    parts = []
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict):
                c = m.get("content") or m.get("text") or ""
                if isinstance(c, list):
                    for ci in c:
                        if isinstance(ci, dict):
                            t = ci.get("text") or ""
                            if isinstance(t, str):
                                parts.append(t)
                        elif isinstance(ci, str):
                            parts.append(ci)
                elif isinstance(c, str):
                    parts.append(c)
            elif isinstance(m, str):
                parts.append(m)
    elif isinstance(msgs, str):
        parts.append(msgs)
    # Fallbacks
    for key in ("problem", "question", "instruction", "prompt"):
        v = entry.get(key)
        if isinstance(v, str) and v:
            parts.append(v)
    text = "\n".join(parts).lower()
    return text

def categorize_camerabench_sft():
    """Load input JSON and count entries per coarse category by keyword match in messages.

    Each entry is assigned to at most ONE category (first-match priority).

    Categories:
    - Translation (Dolly/Pedestal/Truck)
    - Zooming
    - Rotation (Pan/Tilt/Roll)
    - Static / steadiness
    Also prints counts per sub-keyword and lists unmatched entries.
    """
    categories_keywords = {
        "Translation (Dolly/Pedestal/Truck)": ["dolly", "pedestal", "truck"],
        "Zooming": ["zoom", "zoom in", "zoom out"],
        "Rotation (Pan/Tilt/Roll)": ["pan", "tilt", "roll"],
        "Static / steadiness": [
            "static",
            "fixed",
            "stable",
            "stability",
            "shaky",
            "shake",
            "steadiness",
            "shaking",
            "still"
        ],
        "Track": ["track", "follow"],
    }

    def to_pattern(kw: str) -> re.Pattern:
        kw = kw.strip().lower()
        # partial substring match
        return re.compile(re.escape(kw))

    # Preserve insertion order for deterministic first-match priority
    ordered_categories = list(categories_keywords.keys())
    categories_patterns = {
        cat: [to_pattern(kw) for kw in categories_keywords[cat]] for cat in ordered_categories
    }

    data = load_camerabench_sft_json()

    # Flatten entries
    entries = []
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        for key in ("data", "entries", "items", "results"):
            if isinstance(data.get(key), list):
                entries = data[key]
                break
        if not entries:
            entries = [data]

    counts = {cat: 0 for cat in ordered_categories}
    subcounts = {cat: {kw: 0 for kw in categories_keywords[cat]} for cat in ordered_categories}
    unmatched = 0
    unmatched_list = []

    def _entry_label(entry: dict, text: str) -> str:
        for k in ("problem_id", "id", "name", "file", "path"):
            v = entry.get(k)
            if isinstance(v, str) and v:
                return v
        # fallback to truncated text
        t = (text or "").strip().replace("\n", " ")
        return (t[:120] + "...") if len(t) > 120 else t

    for entry in entries:
        text = _extract_text_from_entry(entry)
        assigned_cat = None
        # assign to the first matching category only
        for cat in ordered_categories:
            patterns = categories_patterns[cat]
            matched_cat = False
            for kw, patt in zip(categories_keywords[cat], patterns):
                if patt.search(text):
                    subcounts[cat][kw] += 1
                    matched_cat = True
            if matched_cat:
                counts[cat] += 1
                assigned_cat = cat
                break  # stop after first matched category
        if assigned_cat is None:
            unmatched += 1
            unmatched_list.append(_entry_label(entry, text))

    total_assigned = sum(counts.values()) + unmatched
    print("Category counts (keyword-based, single-category assignment):")
    for cat, cnt in counts.items():
        print(f"- {cat}: {cnt}")
        for kw, kw_cnt in subcounts[cat].items():
            print(f"  * {kw}: {kw_cnt}")
    print(f"- unmatched: {unmatched}")
    print("unmatched:")
    for i, item in enumerate(unmatched_list, 1):
        print(f"  {i}. {item}")
    print(f"Total assigned (matched + unmatched): {total_assigned}")
    print(f"Total entries: {len(entries)}")

    # Ensure all entries sum to 18541
    target_total = 18541
    if total_assigned != target_total:
        print(f"[WARN] Total assigned ({total_assigned}) != expected {target_total}")
    else:
        print(f"[OK] Total assigned equals expected {target_total}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stats and categorization for Camerabench-like datasets")
    parser.add_argument("--input_json", type=str, default="camerabench_sft.json", help="JSON filename or absolute path to read (default: balanced_vqa.json)")
    args = parser.parse_args()

    # Count entries under camerabench_sft directory (unchanged)
    count = count_entries_in_camerabench_sft()
    print(f"Total entries in camerabench_sft: {count}")

    try:
        data = load_json_by_name(args.input_json)
        if isinstance(data, list):
            print(f"Loaded {args.input_json} with {len(data)} entries (list)")
        elif isinstance(data, dict):
            cnt = 1
            for k in ("data", "entries", "items", "results"):
                v = data.get(k)
                if isinstance(v, list):
                    cnt = len(v)
                    break
            print(f"Loaded {args.input_json} (dict), estimated entries: {cnt}")
        else:
            print(f"Loaded {args.input_json} (unknown structure)")
    except Exception as e:
        print(f"Failed to load {args.input_json}: {e}")

    try:
        # Make categorization read the provided input_json
        globals()["load_camerabench_sft_json"] = lambda: load_json_by_name(args.input_json)
        categorize_camerabench_sft()
    except Exception as e:
        print(f"Failed to categorize {args.input_json}: {e}")