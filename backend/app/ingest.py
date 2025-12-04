import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# --------- CONFIG ---------
INPUT_DIR = "scraped_json"
OUTPUT_DIR = "embeddings"
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 64   # best performance/cost tradeoff

os.makedirs(OUTPUT_DIR, exist_ok=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --------- BATCH EMBEDDING FUNCTION ---------
def embed_batch(text_list):
    """
    Sends a batch of texts to OpenAI and returns a list of embedding vectors.
    """
    try:
        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=text_list
        )
        return [d.embedding for d in resp.data]

    except Exception as e:
        print("\nBatch embedding error:", e)
        return [None] * len(text_list)

# --------- MAIN INGESTION LOOP ---------
for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".json"):
        continue

    in_path = os.path.join(INPUT_DIR, filename)
    out_path = os.path.join(OUTPUT_DIR, filename)

    print(f"\nProcessing → {filename}")

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    url = data.get("url", "")
    chunks = data.get("chunks", [])

    output_data = {
        "url": url,
        "chunks": []
    }

    # ---- Batch Process Chunks ----
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        print(f"  Embedding batch {start}–{start+len(batch)-1} (size {len(batch)})")

        vectors = embed_batch(batch)

        # Assign results
        for i, (chunk_text, vec) in enumerate(zip(batch, vectors)):
            chunk_index = start + i
            if vec is None: 
                continue
            output_data["chunks"].append({
                "chunk_index": chunk_index,
                "text": chunk_text,
                "embedding": vec
            })

    # ---- Save embedding file ----
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved embeddings → {out_path}")
