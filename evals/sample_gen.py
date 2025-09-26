import sys
import os
import json

# Add parent directory to path to import main.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import retrieve, chat_completion

def generate_json_output(query_and_meme_list, top_k=3):
    """
    Take a list of strings (query and required meme) and output JSON structure
    
    Args:
        query_and_meme_list (list): List containing [query, required_meme]
        top_k (int): Number of chunks to retrieve
    
    Returns:
        dict: JSON structure with query, required_meme, top_k, and output
    """
    if len(query_and_meme_list) < 2:
        raise ValueError("List must contain at least query and required meme")
    
    query = query_and_meme_list[0]
    required_meme = query_and_meme_list[1]
    
    try:
        # Retrieve relevant chunks
        context = retrieve(query, top_k=top_k)
        
        # Generate model output
        output = chat_completion(query, context)
        
        # Create JSON structure
        result = {
            "query": query,
            "required_meme": required_meme,
            "top_k": top_k,
            "output": output
        }
        
        return result
        
    except Exception as e:
        print(f"Error generating output for query '{query}': {e}")
        return None

def generate_sample(query, required_memes=None, top_k=3):
    """
    Generate a single evaluation sample using retrieve() and chat_completion()
    
    Args:
        query (str): The user query
        required_memes (list, optional): List of required meme names for evaluation
        top_k (int): Number of chunks to retrieve
    
    Returns:
        dict: Sample in the format expected by evals
    """
    try:
        # Retrieve relevant chunks
        context = retrieve(query, top_k=top_k)
        
        # Extract meme names from retrieved chunks
        retrieved_memes = [chunk["meme_name"] for chunk in context]
        
        # Generate model output
        model_output = chat_completion(query, context)
        
        # Create sample dict
        sample = {
            "query": query,
            "retrieved": retrieved_memes,
            "model_output": model_output,
            "context": context  # Include full context for debugging
        }
        
        # Add required_memes if provided
        if required_memes:
            sample["required_memes"] = required_memes
            
        return sample
        
    except Exception as e:
        print(f"Error generating sample for query '{query}': {e}")
        return None

def generate_samples(queries_with_metadata, output_file="samples.jsonl"):
    """
    Generate evaluation samples from a list of queries
    
    Args:
        queries_with_metadata (list): List of dicts with 'query' and optionally 'required_memes'
        output_file (str): Output JSONL file path
    """
    samples = []
    
    for item in queries_with_metadata:
        if isinstance(item, str):
            # Simple string query
            query = item
            required_memes = None
        elif isinstance(item, dict):
            # Dict with query and metadata
            query = item["query"]
            required_memes = item.get("required_memes")
        else:
            print(f"Invalid item format: {item}")
            continue
            
        print(f"Generating sample for: {query}")
        sample = generate_sample(query, required_memes)
        
        if sample:
            samples.append(sample)
        else:
            print(f"Failed to generate sample for: {query}")
    
    # Write to JSONL file
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Generated {len(samples)} samples and saved to {output_file}")
    return samples

if __name__ == "__main__":
    # Example queries - you can manually edit this list
    test_queries = [
        {
            "query": "场景：健身房，用知识库里的'姐姐杀我'，写一个段子。",
            "required_memes": ["姐姐杀我"]
        },
        {
            "query": "场景：健身房，用知识库里的'县城婆罗门'，写一个段子。",
            "required_memes": ["县城婆罗门"]
        },
        {
            "query": "场景：健身房，用知识库里的'顶级过肺'，写一个段子。",
            "required_memes": ["顶级过肺"]
        },
        {
            "query": "场景：健身房，用知识库里的'邪修'，写一个段子。",
            "required_memes": ["邪修"]
        },
        {
            "query": "场景：健身房，用知识库里的'丝瓜汤'，写一个段子。",
            "required_memes": ["丝瓜汤"]
        }
    ]
    
    # Generate samples
    samples = generate_samples(test_queries)
    
    # Print summary
    print("\n" + "="*50)
    print("SAMPLE GENERATION SUMMARY")
    print("="*50)
    for i, sample in enumerate(samples, 1):
        print(f"\nSample {i}:")
        print(f"Query: {sample['query']}")
        print(f"Retrieved memes: {sample['retrieved']}")
        if 'required_memes' in sample:
            print(f"Required memes: {sample['required_memes']}")
        print(f"Model output preview: {sample['model_output'][:100]}...")