import sqlite3
import sqlite_vec

DB_PATH = "vector.db"

conn = sqlite3.connect(DB_PATH)

conn.enable_load_extension(True)
sqlite_vec.load(conn)
conn.enable_load_extension(False)

cur = conn.cursor()

cur.execute("""
CREATE VIRTUAL TABLE IF NOT EXISTS documents USING vec0(
    id INTEGER PRIMARY KEY,
    source TEXT,
    chunk_index INT,
    chunk_text TEXT,
    embedding FLOAT[1536]
);
""")

conn.commit()
conn.close()

print("Vector database initialized!")
