import json

def txt_to_jsonl(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    
    # Split by double newlines to get conversation blocks
    conversation_blocks = content.split('\n\n')
    
    results = []
    
    for block in conversation_blocks:
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        
        if len(lines) < 2:
            continue  # Skip blocks with less than 2 lines
        
        messages = []
        
        # Alternate between user and assistant, starting with user
        for i, line in enumerate(lines):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({
                "role": role,
                "content": line
            })
        
        # Only add if we have at least one complete user-assistant pair
        if len(messages) >= 2:
            results.append({
                "messages": messages
            })

    # Write as JSONL (one object per line)
    with open(output_path, "w", encoding="utf-8") as f:
        for obj in results:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"✅ Converted {len(results)} conversation blocks to {output_path}")

# Example usage:
txt_to_jsonl("transcription/transcription_zhangzhiqiang_500lines.txt", "knowledge_base/fine_tune_pun.jsonl")
