import os
import sqlite3
import sqlite_vec
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
# --- Configuration ---
DB_PATH = "vector.db"
TOP_K = 3  # number of top chunks to retrieve

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Connect to SQLite and enable vector extension
conn = sqlite3.connect(DB_PATH)
conn.enable_load_extension(True)
sqlite_vec.load(conn)
conn.enable_load_extension(False)
cur = conn.cursor()


def get_top_chunks(question, top_k=TOP_K):
    """
    Embed the user question and retrieve the top_k most similar chunks.
    """
    # 1. Embed the question
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )
    question_vector = response.data[0].embedding

    # 2. Query SQLite for closest vectors using cosine similarity
    cur.execute("""
        SELECT id, vec_distance_cosine(embedding, ?) AS score
        FROM documents
        ORDER BY score ASC
        LIMIT ?;
    """, (sqlite_vec.serialize_float32(question_vector), TOP_K))



    return cur.fetchall()


def answer_question(question):
    """
    Returns a GPT-generated answer using retrieved chunks as context.
    """
    top_chunks = get_top_chunks(question)

    if not top_chunks:
        return "Sorry, I couldn't find relevant information."

    # Build context
    context_text = "\n\n".join([f"Source: {src}\n{txt}" for src, txt in top_chunks])

    # Prompt GPT with context + user question
    prompt = f"""
    You are a helpful assistant. Use the following context to answer the question.
    
    Context:
    {context_text}
    
    Question: {question}
    Answer:
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()


# --- Example usage ---
if __name__ == "__main__":
    while True:
        user_input = input("Ask a question (or type 'quit' to exit): ")
        if user_input.lower() in ("quit", "exit"):
            break
        answer = answer_question(user_input)
        print("\nAnswer:", answer, "\n")
