Here’s a clean `README.md` draft that explains the full architecture and code workflow we built for your **Meme RAG** project:

---

# Meme RAG: Retrieval-Augmented Generation for Internet Memes

This project implements a **RAG (Retrieval-Augmented Generation) system for internet memes**, where each meme is stored with a **name**, **definition**, and **usage examples** (rated by funniness).
The goal is to retrieve the correct meme context even when meme names, definitions, and usage examples are **not semantically aligned**.

---

## 🚀 Features

* **Structured Chunking**: Splits memes into `definition` and `usage` chunks, preserving metadata like `meme_name`, `type`, and `funny_rating`.
* **Vector Search (FAISS)**: Uses **SentenceTransformer embeddings** for semantic retrieval.
* **Metadata Filtering**: Ensures precise control (e.g., retrieve only definitions, only a given meme’s usages, or rank by funniness).
* **Hybrid Retrieval**: Combines semantic search with structured filtering to avoid confusion between unrelated fields.

---

## 📂 Data Format

Memes are stored as JSON:

```json
{
  "meme_name": "你怎么脑袋尖尖的",
  "definition": "‘你怎么脑袋尖尖的’起源于博主常熟阿诺，用于形容一个人脑子笨笨的，常见于健身健美人群。可以单独使用或者缩减为‘脑袋尖尖的’",
  "update_time": "2025-09-25",
  "usages": [
    {"phrase": "你怎么脑袋尖尖的", "class": "独立", "funny_rating": 3},
    {"phrase": "脑袋尖尖的", "class": "形容词", "funny_rating": 3}
  ]
}

{
  "meme_name": "真南",
  "definition": "‘南’与‘难’念谐音。代表我面临好多困难。",
  "update_time": "2025-09-25",
  "usages": [
    {"phrase": "南", "class": "形容词", "funny_rating": 3},
    {"phrase": "好难啊", "class": "独立", "funny_rating": 3}
  ]
}

{
  "meme_name": "沪爷",
  "definition": "‘沪爷’指上海的富二代，也是调侃在上海的人。",
  "update_time": "2025-09-25",
  "usages": [
    {"phrase": "沪爷", "class": "名词（人）", "funny_rating": 3}
  ]
}

{
  "meme_name": "广式双马尾",
  "definition": "‘广式双马尾’指蟑螂。",
  "update_time": "2025-09-25",
  "usages": [
    {"phrase": "广东双马尾", "class": "名词（物）", "funny_rating": 3}
  ]
}
```

---

## 🧩 Chunking Strategy

Each meme is split into multiple **chunks** for indexing:

```json
[
  {
    "id": "Doge_definition",
    "meme_name": "Doge",
    "type": "definition",
    "content": "Doge: A meme featuring a Shiba Inu dog with captions in broken English."
  },
  {
    "id": "Doge_usage_0",
    "meme_name": "Doge",
    "type": "usage",
    "funny_rating": 5,
    "content": "Doge (usage): Such wow, much fun."
  },
  {
    "id": "Doge_usage_1",
    "meme_name": "Doge",
    "type": "usage",
    "funny_rating": 4,
    "content": "Doge (usage): Very science, so smart."
  }
]
```

This ensures that **definitions and usages are retrievable independently**, while still linked to their parent meme.

---

## ⚙️ Workflow

1. **Load dataset** of memes in JSON.
2. **Chunk** into field-based entries (definition + usage).
3. **Embed** chunks using [SentenceTransformers](https://www.sbert.net/).
4. **Store** embeddings in a FAISS index.
5. **Retrieve** chunks with:

   * Semantic similarity search
   * Metadata filters (e.g., only `type=definition`)
   * Optional reranking by `funny_rating`

---

## 🖥️ Example Code

### Retrieval Function

```python
def retrieve(query, top_k=3, filter_metadata=None, funny_sort=False):
    query_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = index.search(query_vec, top_k * 5)

    results = []
    for score, idx in zip(scores[0], idxs[0]):
        chunk = all_chunks[idx].copy()
        chunk["score"] = float(score)
        results.append(chunk)

    # Metadata filtering
    if filter_metadata:
        for key, value in filter_metadata.items():
            results = [r for r in results if r.get(key) == value]

    # Sort usages by funniness if requested
    if funny_sort:
        results = sorted(results, key=lambda x: (x["funny_rating"] or 0), reverse=True)

    return results[:top_k]
```

### Example Queries

```python
# Get a meme definition
retrieve("What is doge?", filter_metadata={"type": "definition"})

# Get funny examples of shiba inu memes
retrieve("Shiba inu meme funny example", filter_metadata={"type": "usage"}, funny_sort=True)

# Get the funniest Rickroll usage
retrieve("funniest Rickroll usage", filter_metadata={"meme_name": "Rickroll", "type": "usage"}, funny_sort=True)
```

---

## 🔍 Why Metadata Filtering?

* **BM25 alone** works well for **exact name matches** (e.g., "doge meme"), but fails for semantic queries like "shiba inu meme".
* **Embeddings alone** capture semantic meaning, but can confuse definitions with usage examples.
* ✅ **Hybrid approach** (embeddings + metadata filters + reranking) ensures accurate retrieval:

  * `meme_name` matches → highest priority
  * `definition` → used for "what is" style queries
  * `usage` → used for "example" or "funniest" style queries

---

## 📊 Architecture Overview

```text
           ┌────────────┐
           │   Dataset  │
           │ (memes)    │
           └─────┬──────┘
                 │
         Structured Chunking
                 │
           ┌─────▼──────┐
           │ Embeddings │  ← SentenceTransformers
           └─────┬──────┘
                 │
          ┌──────▼───────┐
          │ FAISS Index  │
          └──────┬───────┘
                 │
      ┌──────────▼──────────┐
      │  Retrieval Engine   │
      │  - Semantic Search  │
      │  - Metadata Filter  │
      │  - Reranking        │
      └─────────┬───────────┘
                │
         ┌──────▼──────┐
         │   Results   │
         └─────────────┘
```

---

## 📌 Next Steps

* Build a **chat interface** to test meme Q\&A interactively.
* Use **rerankers (cross-encoders)** for even better precision.