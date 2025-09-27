import json

def txt_to_jsonl(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    results = []
    current_stem = None
    examples = []

    for line in lines + [""]:  # add sentinel empty line
        if line == "":
            if current_stem and examples:
                for ex in examples:
                    results.append({
                        "messages": [
                            {"role": "user", "content": current_stem},
                            {"role": "assistant", "content": ex}
                        ]
                    })
            # reset for next stem
            current_stem = None
            examples = []
        else:
            if current_stem is None:
                current_stem = line
            else:
                examples.append(line)

    # write as JSONL (one object per line)
    with open(output_path, "w", encoding="utf-8") as f:
        for obj in results:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"✅ Converted {len(results)} (stem, example) pairs to {output_path}")

# Example usage:
# txt_to_jsonl("raw_data.txt", "fine_tune_data.jsonl")
