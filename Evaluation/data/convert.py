import os
import json

def convert_binary_classification_to_camerabench_binary():
    """
    Read all JSON files under ./binary_classification, convert each entry to Camerabench binary format,
    and save all converted entries into ../camerabench_binary.json (created if missing).
    """
    base_dir = os.path.dirname(__file__)
    src_dir = os.path.join(base_dir, "binary_classification")
    out_path = os.path.join(os.path.dirname(base_dir), "camerabench_binary.json")

    if not os.path.isdir(src_dir):
        print(f"Source directory not found: {src_dir}")
        return

    results = []
    problem_id = 1

    def convert_entry(e: dict) -> dict:
        image = e.get("image", "")
        question = e.get("question", "")
        label = (e.get("label") or "").strip().lower()
        # Map label to solution letter
        if label == "yes":
            solution = "<answer>B</answer>"
        elif label == "no":
            solution = "<answer>A</answer>"
        else:
            solution = "<answer>A</answer>"
        return {
            "problem_id": None,  # fill later
            # 使用原始的问题
            "problem": question,
            "data_type": "video",
            "problem_type": "multiple choice",
            # 选项固定为 Yes/No
            "options": [
                "A. No",
                "B. Yes"
            ],
            "process": "",
            "solution": solution,
            "path": f"video/{image}" if image else "",
            "data_source": "camerabench",
            "dataset": "camerabench",
            "output": []
        }

    # Iterate all json or jsonl files inside src_dir
    for fname in sorted(os.listdir(src_dir)):
        if not (fname.endswith(".json") or fname.endswith(".jsonl")):
            continue
        fpath = os.path.join(src_dir, fname)
        # 以源文件名(不含扩展名)作为 id 前缀
        id_prefix = os.path.splitext(fname)[0]
        entries = []
        if fname.endswith(".jsonl"):
            # 逐行解析 JSON Lines 格式
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            if isinstance(obj, dict):
                                entries.append(obj)
                        except Exception as ex_line:
                            print(f"Skip line in {fname}: {ex_line}")
            except Exception as ex:
                print(f"Skip {fpath}: {ex}")
                continue
        else:
            # 普通 JSON 文件
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as ex:
                print(f"Skip {fpath}: {ex}")
                continue
            if isinstance(data, list):
                entries = data
            elif isinstance(data, dict):
                # common keys that may contain list of entries
                for key in ("data", "entries", "items", "results"):
                    if isinstance(data.get(key), list):
                        entries = data[key]
                        break
                if not entries:
                    entries = [data]
            else:
                print(f"Unrecognized data format in {fpath}, skipping")
                continue

        file_count = 0
        local_index = 1
        for e in entries:
            if not isinstance(e, dict):
                continue
            converted = convert_entry(e)
            # 使用源文件名作为 problem_id 前缀
            converted["problem_id"] = f"{id_prefix}_{problem_id}"
            # 保留 id 字段，使用源文件名作为前缀
            converted["id"] = f"{id_prefix}_{local_index}"
            results.append(converted)
            problem_id += 1
            local_index += 1
            file_count += 1
        print(f"Processed file: {fname}, entries: {file_count}")

    # Write output
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(results)} entries to {out_path}")


if __name__ == "__main__":
    convert_binary_classification_to_camerabench_binary()