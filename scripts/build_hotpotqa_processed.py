import json
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_list_of_strings(obj, field_name: str):
    if not isinstance(obj, list):
        raise ValueError(f"{field_name} 必须是 list，实际类型为 {type(obj)}")
    result = []
    for i, x in enumerate(obj):
        if not isinstance(x, str):
            raise ValueError(f"{field_name}[{i}] 必须是 str，实际类型为 {type(x)}")
        result.append(x)
    return result


def normalize_supporting_facts(supporting_facts):
    """
    统一输出为:
    [
      [title, sent_id],
      ...
    ]
    """
    if isinstance(supporting_facts, dict):
        titles = supporting_facts.get("title", [])
        sent_ids = supporting_facts.get("sent_id", [])
        if len(titles) != len(sent_ids):
            raise ValueError("supporting_facts 中 title 与 sent_id 长度不一致")
        return [[str(t), int(sid)] for t, sid in zip(titles, sent_ids)]

    if isinstance(supporting_facts, list):
        normalized = []
        for i, item in enumerate(supporting_facts):
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError(f"supporting_facts[{i}] 格式非法，应为 [title, sent_id]")
            title, sent_id = item
            normalized.append([str(title), int(sent_id)])
        return normalized

    raise ValueError(f"无法识别的 supporting_facts 格式: {type(supporting_facts)}")


def normalize_context(context):
    """
    统一输出为:
    [
      {
        "title": "...",
        "sentences": ["句子1", "句子2"]
      }
    ]
    """
    if isinstance(context, dict):
        titles = context.get("title", [])
        sentences = context.get("sentences", [])

        if len(titles) != len(sentences):
            raise ValueError("context 中 title 与 sentences 长度不一致")

        normalized = []
        for title, sents in zip(titles, sentences):
            normalized.append({
                "title": str(title),
                "sentences": ensure_list_of_strings(sents, f"context[{title}].sentences")
            })
        return normalized

    if isinstance(context, list):
        normalized = []
        for i, item in enumerate(context):
            if not isinstance(item, dict):
                raise ValueError(f"context[{i}] 必须是 dict")
            if "title" not in item or "sentences" not in item:
                raise ValueError(f"context[{i}] 缺少 title 或 sentences 字段")

            normalized.append({
                "title": str(item["title"]),
                "sentences": ensure_list_of_strings(item["sentences"], f"context[{i}].sentences")
            })
        return normalized

    raise ValueError(f"无法识别的 context 格式: {type(context)}")


def get_qid(sample: dict):
    if "qid" in sample:
        return str(sample["qid"])
    if "_id" in sample:
        return str(sample["_id"])
    if "id" in sample:
        return str(sample["id"])
    raise ValueError("样本中缺少 qid / _id / id 字段，无法生成 qid")


def normalize_sample(sample: dict):
    required_fields = [
        "question",
        "answer",
        "type",
        "level",
        "supporting_facts",
        "context"
    ]
    for field in required_fields:
        if field not in sample:
            raise ValueError(f"样本缺少必要字段: {field}")

    normalized = {
        "qid": get_qid(sample),
        "question": str(sample["question"]),
        "answer": str(sample["answer"]),
        "type": str(sample["type"]),
        "level": str(sample["level"]),
        "supporting_facts": normalize_supporting_facts(sample["supporting_facts"]),
        "context": normalize_context(sample["context"]),
    }
    return normalized


def convert_file(raw_path: Path, processed_path: Path):
    data = load_json(raw_path)
    if not isinstance(data, list):
        raise ValueError(f"{raw_path} 顶层必须是 list")

    processed_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with processed_path.open("w", encoding="utf-8") as fout:
        for idx, sample in enumerate(data):
            try:
                normalized = normalize_sample(sample)
            except Exception as e:
                raise ValueError(f"处理样本失败，文件={raw_path}，索引={idx}，错误={e}") from e

            fout.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            count += 1

    return count


def main():
    project_root = Path(__file__).resolve().parent.parent

    base_dir = project_root / "data" / "hotpotqa_distractor"
    raw_dir = base_dir / "raw"
    processed_dir = base_dir / "processed"

    train_raw = raw_dir / "train.json"
    val_raw = raw_dir / "val.json"
    test_raw = raw_dir / "test.json"

    train_processed = processed_dir / "train.jsonl"
    val_processed = processed_dir / "val.jsonl"
    test_processed = processed_dir / "test.jsonl"

    for path in [train_raw, val_raw, test_raw]:
        if not path.exists():
            raise FileNotFoundError(f"找不到原始切分文件: {path}")

    processed_dir.mkdir(parents=True, exist_ok=True)

    train_count = convert_file(train_raw, train_processed)
    val_count = convert_file(val_raw, val_processed)
    test_count = convert_file(test_raw, test_processed)

    print("HotpotQA 标准化完成：")
    print(f"  train -> {train_processed}  ({train_count} 条)")
    print(f"  val   -> {val_processed}    ({val_count} 条)")
    print(f"  test  -> {test_processed}   ({test_count} 条)")


if __name__ == "__main__":
    main()