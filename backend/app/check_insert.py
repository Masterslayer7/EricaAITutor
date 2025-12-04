import sqlite3
import sqlite_vec

DB_PATH = "vector.db"

conn = sqlite3.connect(DB_PATH)
conn.enable_load_extension(True)
sqlite_vec.load(conn)
conn.enable_load_extension(False)
cur = conn.cursor()

# Count total rows
cur.execute("SELECT COUNT(*) FROM documents")
print(f"Total documents: {cur.fetchone()[0]}")

# Show first few rows (without embedding to keep output readable)
cur.execute("""
    SELECT id, source, chunk_index, 
           SUBSTR(chunk_text, 1, 100) as text_preview,
           LENGTH(embedding) as embedding_size
    FROM documents 
    LIMIT 5
""")

print("\nFirst 5 rows:")
for row in cur.fetchall():
    print(f"ID: {row[0]}")
    print(f"Source: {row[1]}")
    print(f"Chunk: {row[2]}")
    print(f"Text: {row[3]}...")
    print(f"Embedding size (bytes): {row[4]}")
    print("-" * 80)

conn.close()