import os
import json
import glob
import logging
import ollama
import asyncio
import re
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash
from nano_graphrag.prompt import PROMPTS
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# We overwrite the default prompt to force the LLM to find your specific 
# node types (Concept, Resource, Example) and edge types (prereq_of, etc.)
ERICA_TUPLE_PROMPT = """
-Goal-
You are an expert AI Tutor system named Rebeca.
Analyze the content below and extract the pedagogical Knowledge Graph.

-Nodes to Identify-
1. Concept: Abstract ideas (e.g., Jensen's Inequality).
2. Resource: Explicit content references (videos, slides).
3. Example: Worked problems.

-Relationships to Identify-
1. prereq_of: (Concept A) is a prerequisite for (Concept B)
2. explains: (Resource) explains (Concept)
3. exemplifies: (Example) exemplifies (Concept)
4. near_transfer: (Concept A) is related/similar to (Concept B)

-Output Format-
Output strictly one entry per line.
IMPORTANT: 
1. Use the delimiter "<|>" exactly. 
2. DO NOT add spaces around the delimiter.
3. DO NOT add quotes around the Name unless they are part of the text.

Correct Format:
("entity"<|>Name<|>Type<|>Description)
("relationship"<|>Source<|>Target<|>RelationshipType)

Example Output:
("entity"<|>Automated Reasoning<|>concept<|>difficulty: intermediate; def: A method used in AI...)
("entity"<|>Lecture 1<|>resource<|>url: http://...)
("relationship"<|>Automated Reasoning<|>Knowledge Base<|>relates_to)

Content to Analyze:
{input_text}
"""

# This updates the prompt globally for this script execution
PROMPTS["entity_extraction"] = ERICA_TUPLE_PROMPT

MODEL = "gpt-4o-mini"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "scraped_json")
WORKING_DIR = "./erica_graph_storage"
json_files = ["/home/yugp/projects/EricaAITutor/backend/scripts/scraped_json/automated-reasoning.json"]

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

# Code taken and modified from https://github.com/gusye1234/nano-graphrag/blob/main/examples/using_ollama_as_llm.py
async def openai_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    client = AsyncOpenAI() # Automatically picks up env variable
    
    # Cleanup args that might crash OpenAI
    kwargs.pop("max_tokens", None)
    kwargs.pop("response_format", None)
    hashing_kv = kwargs.pop("hashing_kv", None)

    messages = []
    if system_prompt: messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # Cache Check
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        cached = await hashing_kv.get_by_id(args_hash)
        if cached: return cached["return"]

    # Call GPT-4o
    print("⏳ Calling OpenAI GPT-4o...")
    response = await client.chat.completions.create(
        model=MODEL, messages=messages, **kwargs
    )
    result = response.choices[0].message.content
    cleaned = result.replace("json", "").replace("", "").strip()
    
    data = json.loads(cleaned)
    
    # Cache Save
    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": data, "model": MODEL}})

    return result

# --- 3. EXECUTION ---
if __name__ == "__main__":
    # RESET STORAGE
    if os.path.exists(WORKING_DIR):
        import shutil
        shutil.rmtree(WORKING_DIR)

    rag = GraphRAG(
        working_dir=WORKING_DIR,
        enable_llm_cache=True,
        best_model_func=openai_model_func,
        cheap_model_func=openai_model_func,
    )

    # Manual Single File Load
    with open(json_files[0], "r") as f:
        data = json.load(f)
        content = data.get("text", "") or "\n".join([c["text"] for c in data["chunks"]])
        
    print(f"Processing: {json_files[0]}...")
    rag.insert(content)
    
    print("✅ Done! Check rebeca_graph_storage for graphml file.")