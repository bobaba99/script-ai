import faiss
import os
from dotenv import load_dotenv
import numpy as np
from openai import OpenAI
import json
from volcenginesdkarkruntime import Ark

# TOKENIZERS_PARALLELISM=false # To suppress tokenizer parallelism warning in .env
load_dotenv()

# Model names
OPENAI_MODEL_NAME = "text-embedding-ada-002"
OPENAI_CHAT_MODEL = "ft:gpt-4.1-2025-04-14:personal::CKgaEPPq"  # ft:gpt-4.1-2025-04-14:personal::CKgaEPPq
ARK_CHAT_MODEL = "ep-20251001101741-9cx8r" # manual set up in Ark dashboard 在线推理-自定义推理接入点-创建推理接入点-模型仓库
MODEL_SELECTED = "ark"

# Load system prompt from external file
with open('prompts/developer_prompt.md', 'r', encoding='utf-8') as f:
    developer_prompt = f.read().strip()

# Load user prompt template from external file
with open('prompts/user_prompt.md', 'r', encoding='utf-8') as f:
    user_prompt_template = f.read().strip()
    

# -------------------------
# Embedding functions
# -------------------------
model_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# model_ark = Ark(api_key=os.getenv("ARK_API_KEY"), base_url="https://ark.cn-beijing.volces.com/api/v3")
model_ark = OpenAI(base_url="https://ark.cn-beijing.volces.com/api/v3", api_key=os.environ.get("ARK_API_KEY"))

def embed_openai(texts):
    """
    Embed a list of texts using OpenAI ada v2.
    Returns a numpy array of shape (len(texts), 1536).
    """
    response = model_openai.embeddings.create(
        input=texts,
        model=OPENAI_MODEL_NAME,
        encoding_format="float"
    )
    vecs = np.array([item.embedding for item in response.data], dtype=np.float32)
    # L2-normalise for cosine via inner product
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    vecs = vecs / norms
    return np.ascontiguousarray(vecs)

# -------------------------
# 1. Structured Chunking
# -------------------------
def chunking(json_obj):
    """
    Chunk a meme JSON object into smaller pieces with metadata.
    Each chunk is a dict with keys:
    id, meme_name, type (definition/usage), funny_rating, content, tags, class.
    """
    name = json_obj["meme_name"]
    
    # Skip entries with empty meme_name
    if not name.strip():
        return []
    
    chunks = [{
        "id": f"{name}_definition",
        "meme_name": name,
        "type": "definition", 
        "update_time": json_obj.get("update_time", ""),
        "tags": json_obj.get("tags", []),
        "funny_rating": None,
        "class": json_obj.get("class", []),
        "phrase": None,
        "content": f"{name}: {json_obj['definition']}"
    }]
    
    for i, usage in enumerate(json_obj["usages"]):
        # Skip usages with empty phrases or negative funny_rating
        if not usage.get("phrase") or usage.get("funny_rating", 0) < 0:
            continue
            
        chunks.append({
            "id": f"{name}_usage_{i}",
            "meme_name": name,
            "type": "usage",
            "update_time": json_obj.get("update_time", ""),
            "tags": json_obj.get("tags", []),
            "funny_rating": usage["funny_rating"],
            "phrase": usage["phrase"],
            "class": usage["class"],
            "content": f"{name} (usage, {usage['class']}): {usage['phrase']}"
        })
    
    return chunks

# -------------------------
# Load and chunk dataset, one time
# Load dataset from knowledge_base.json
# with open('/knowledge_base/raw_memes.json', 'r', encoding='utf-8') as f:
#     dataset = json.load(f)

# # Ensure dataset is a list
# if not isinstance(dataset, list):
#     dataset = [dataset]

# all_chunks = [chunk for meme in dataset for chunk in chunking(meme)]

# # Save chunked data to knowledge_base_chunk.json
# with open('/knowledge_base/knowledge_base_chunk.json', 'w', encoding='utf-8') as f:
#     json.dump(all_chunks, f, ensure_ascii=False, indent=2)
# -------------------------

# -------------------------
# 2. Embeddings + FAISS Index
# -------------------------
def build_index(texts: list[str]):
    embs = embed_openai(texts)
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    try:
        index.add(embs) # type: ignore
    except Exception as e:
        print(f"Error adding embeddings to index: {e}")
        print(f"Embeddings shape: {embs.shape}, dtype: {embs.dtype}")
        raise
    return index, embs

with open('knowledge_base/knowledge_base_chunk.json', 'r', encoding='utf-8') as f:
    all_chunks = json.load(f)


texts = [chunk["content"] for chunk in all_chunks]
index, embeddings = build_index(texts)


# -------------------------
# 3. Hybrid Retrieval
# -------------------------
def retrieve(query: str, top_k: int = 3, **filters):
    """
    Retrieve top_k relevant chunks for the query with flexible metadata filtering.
    
    Args:
        query: Search query string
        top_k: Number of results to return
        **filters: Flexible metadata filters using keyword arguments:
            - type: Filter by type ("definition" or "usage")  
            - meme_name: Filter by meme name(s) - string or list
            - funny_rating: Filter by minimum funny rating
            - date: Filter by specific date
            - funny_sort: Sort by funny_rating (True/False, default False)
    
    Returns:
        List of dicts containing content, metadata, and similarity scores
    
    Example usage:
        # Basic search
        retrieve("funny meme")
        
        # Search only definitions
        retrieve("what is rickroll", type="definition")
        
        # Search specific memes with usage examples
        retrieve("funny usage", meme_name="Rickroll", type="usage")
        
        # Search multiple memes, sorted by funny rating
        retrieve("humor", meme_name=["Rickroll", "Doge"], funny_sort=True)
        
        # High-quality content only
        retrieve("best memes", funny_rating=4, funny_sort=True)
        
        # Search with date filter
        retrieve("recent memes", date="2025-01-15", top_k=5)
    """
    try:
        # Extract special parameters
        funny_sort = filters.pop('funny_sort', False)
        
        q = embed_openai([query])
        search_k = min(top_k * 7, index.ntotal)  # Don't search for more than available
        D, I = index.search(q, search_k)  # type: ignore
        results = []
        
        for score, idx in zip(D[0], I[0]):
            if idx == -1 or idx >= len(all_chunks):
                continue
            item = all_chunks[idx].copy()
            item["score"] = float(score)
            results.append(item)

        # Flexible metadata filtering
        for key, value in filters.items():
            if isinstance(value, list):
                # Handle list values (e.g., meme_name=["Rickroll", "Doge"])
                results = [r for r in results if r.get(key) in value]
            elif key == "funny_rating" and isinstance(value, (int, float)):
                # Handle minimum funny rating filter
                results = [r for r in results if (r.get(key) or 0) >= value]
            else:
                # Handle exact match
                results = [r for r in results if r.get(key) == value]

        # Optional rerank by funny rating
        if funny_sort:
            results.sort(key=lambda x: (x.get("funny_rating") or 0), reverse=True)

        return results[:top_k]
    except Exception as e: 
        print(f"Error in retrieve function: {e}")
        return []

# -------------------------
# 4. Chat Completion
# -------------------------

def chat_completion(user_query: str, context: list[dict], model="openai"):
    # Format retrieved chunks for the template
    formatted_chunks = []
    for chunk in context:
        chunk_info = f"meme_name: {chunk['meme_name']}\n"
        chunk_info += f"definition: {chunk.get('content', '')}\n"
        if chunk.get('phrase'):
            chunk_info += f"phrase: {chunk['phrase']}\n"
        if chunk.get('class'):
            chunk_info += f"class: {chunk['class']}\n"
        if chunk.get('funny_rating') is not None:
            chunk_info += f"funny_rating: {chunk['funny_rating']}\n"
        formatted_chunks.append(chunk_info)
    
    retrieved_chunks_text = "\n---\n".join(formatted_chunks)
    
    # Fill in the template
    user_message = user_prompt_template.format(
        retrieved_chunks_here=retrieved_chunks_text,
        user_query=user_query
    )
    
    if model == "openai":
        response = model_openai.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {"role": "developer", "content": developer_prompt},
                {"role": "user", "content": user_message}
            ]
        )
    elif model == "ark":
        response = model_ark.chat.completions.create(
            model=ARK_CHAT_MODEL,
            messages=[
                {"role": "system", "content": developer_prompt},
                {"role": "user", "content": user_message}
            ]
        )
    else:
        raise ValueError(f"Unsupported model: {model}")

    return response.choices[0].message.content # type: ignore

def generate_json_output(query_list, top_k=3):
    """
    Take a JSON list of objects and output JSON structure for each
    
    Args:
        query_list (list): List of dicts with "query" and "required_meme" keys
                          [{"query": "", "required_meme": ""}, ...]
        top_k (int): Number of chunks to retrieve
    
    Returns:
        list: List of JSON structures with query, required_meme, top_k, and output
    """
    if not query_list:
        raise ValueError("List cannot be empty")
    
    results = []
    
    for item in query_list:
        if not isinstance(item, dict) or "query" not in item or "required_meme" not in item:
            print(f"Invalid item format: {item}. Expected dict with 'query' and 'required_meme' keys")
            continue
            
        query = item["query"]
        meme = item["required_meme"]
        
        try:
            # Retrieve relevant chunks
            context = retrieve(query, top_k=top_k)
            
            # Generate model output
            output = chat_completion(query, context, model=MODEL_SELECTED)
            
            # Create JSON structure
            result = {
                "query": query,
                "required_meme": meme,
                "top_k": top_k,
                "output": output
            }
            
            results.append(result)
            print(f"Query: {query}")
            
        except Exception as e:
            print(f"Error generating output for query '{query}': {e}")
            continue
    
    return results

query_list = [
    {"query": "场景在健身房，用知识库里的'邪修'，写一个关于博主艾尔加朵的段子。", "required_meme": "邪修"},
    {"query": "场景在健身房，用知识库里的'丝瓜汤'，写一个关于博主艾尔加朵的段子。", "required_meme": "丝瓜汤"},
    {"query": "场景在健身房，用知识库里的'美美桑内'，写一个关于博主艾尔加朵的段子。", "required_meme": "美美桑内"},
    {"query": "场景在健身房，用知识库里的'顶级过肺'，写一个关于博主艾尔加朵的段子。", "required_meme": "顶级过肺"},
    {"query": "场景在健身房，用知识库里的'县城婆罗门'，写一个关于博主艾尔加朵的段子。", "required_meme": "县城婆罗门"},
    {"query": "场景在健身房，用知识库里的'被做局了'，写一个关于博主艾尔加朵的段子。", "required_meme": "被做局了"},
    {"query": "场景在健身房，用知识库里的'癫公癫婆'，写一个关于博主艾尔加朵的段子。", "required_meme": "癫公癫婆"},
    {"query": "场景在健身房，用知识库里的'姐姐杀我'，写一个关于博主艾尔加朵的段子。", "required_meme": "姐姐杀我"},
]
eval_result = generate_json_output(query_list, top_k=3)

with open('samples.json', 'w', encoding='utf-8') as f:
    json.dump(eval_result, f, ensure_ascii=False, indent=2)