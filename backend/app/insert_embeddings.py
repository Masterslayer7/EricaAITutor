import os
import json
import sqlite3
import sqlite_vec

DB_PATH = "vector.db"
EMBED_DIR = "embeddings"

# Connect
conn = sqlite3.connect(DB_PATH)

# Enable extension load
conn.enable_load_extension(True)
sqlite_vec.load(conn)
conn.enable_load_extension(False)

cur = conn.cursor()

# Iterate over all JSON files
for filename in os.listdir(EMBED_DIR):
    if not filename.endswith(".json"):
        continue

    fpath = os.path.join(EMBED_DIR, filename)

    with open(fpath, "r") as f:
        data = json.load(f)

    url = data["url"]
    for chunk in data["chunks"]:
        chunk_index = chunk["chunk_index"]
        chunk_text = chunk["text"]
        embedding = chunk["embedding"]

        # Convert embedding list to sqlite_vec vector
        embedding_bytes = sqlite_vec.serialize_float32(embedding)  # note lowercase 'vector'

        cur.execute("""
            INSERT INTO documents (source, chunk_index, chunk_text, embedding)
            VALUES (?, ?, ?, ?)
        """, (url, chunk_index, chunk_text, embedding_bytes))

conn.commit()
conn.close()

print("Inserted all chunks into SQLite!")
