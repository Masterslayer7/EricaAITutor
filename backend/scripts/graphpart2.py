import os
import json
import re
import logging
import shutil
import datetime
from dotenv import load_dotenv
from openai import AsyncOpenAI
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash
from nano_graphrag.prompt import PROMPTS

# --- CONFIGURATION ---
load_dotenv()
MODEL = "gpt-4o-mini"
WORKING_DIR = "./rebeca_graph_storage"
TEST_FILE = "/home/yugp/projects/EricaAITutor/backend/scripts/scraped_json/automated-reasoning.json"
DEBUG_FILE = "debug_trace.log"


logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

# --- 1. PROMPT (Standard JSON) ---
# We ask for the exact JSON structure nano-graphrag looks for keys in.
ERICA_JSON_PROMPT = """
You are an expert AI Tutor system named Rebeca.
Analyze the content below and extract the pedagogical Knowledge Graph.

-Nodes to Identify-
1. Concept: Abstract ideas.
2. Resource: Explicit content references.
3. Example: Worked problems.

-Relationships to Identify-
1. prereq_of: (Concept A) is a prerequisite for (Concept B)
2. explains: (Resource) explains (Concept)
3. exemplifies: (Example) exemplifies (Concept)
4. near_transfer: (Concept A) is related/similar to (Concept B)

Return a single valid JSON object. No other text
Example:
{{
  "entities": [
    {{
      "name": "string",
      "type": "concept | resource | example",
      "difficulty": "basic | intermediate | advanced",
      "definition": "string"
    }}
  ],
  "relationships": [
    {{
      "source": "string",
      "target": "string",
      "relationship": "prereq_of | explains | near_transfer | exemplifies" 
    }}
    ]
}}

Content to Analyze:
{input_text}
"""
PROMPTS["entity_extraction"] = ERICA_JSON_PROMPT

# def dict_to_rag_string(data: dict) -> str:
#     lines = []
#     valid_nodes = set()
    
#     for node in data.get("entities", []):
#         name = node.get("name", "Unknown").strip().upper()
#         type_ = node.get("type", "concept")
#         desc = node.get("definition", "No definition").replace("\n", " ").replace('"', "'")
#         lines.append(f'("entity"<|>{name}<|>{type_}<|>{desc})')
#         valid_nodes.add(name)

#     for edge in data.get("relationships", []):
#         src = edge.get("source", "Unknown").strip().upper()
#         tgt = edge.get("target", "Unknown").strip().upper()
#         rel = edge.get("relationship", "related_to")
#         if src in valid_nodes and tgt in valid_nodes:
#             lines.append(f'("relationship"<|>{src}<|>{tgt}<|>{rel})')

#     return "\n".join(lines)

def log_debug(stage, message):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] [{stage}] {message}"
    print(entry)
    with open(DEBUG_FILE, "a", encoding="utf-8") as f:
        f.write(entry + "\n")
        
with open(DEBUG_FILE, "w") as f:
    f.write("--- ERICA DEBUG SESSION STARTED ---\n")

# --- 2. CLEANER FUNCTION (The Middleware) ---
# This is what you identified: a custom parser to sanitize the LLM output
def manual_clean_and_parse(response: str) -> dict:
    """
    1. Strips markdown.
    2. Parses JSON.
    3. Normalizes all Names to UPPERCASE (Crucial for Edge matching).
    """
    log_debug("CLEANER", "Received raw string from LLM. Length: " + str(len(response)))
    
    try:
        # Strip Markdown
        clean = response.replace("```json", "").replace("```", "").strip()
        clean = re.sub(r",\s*}", "}", clean) # Fix trailing commas
        clean = re.sub(r",\s*]", "]", clean)
        
        data = json.loads(clean)
        
        # --- NORMALIZATION STEP (Fixes "0 Edges" issues) ---
        normalized = {"entities": [], "relationships": []}
        valid_nodes = set()
        
        
        raw_entities = data.get("entities", [])
        log_debug("CLEANER", f"Found {len(raw_entities)} raw entities in JSON.")

        # 1. Normalize Entities
        for i, node in data.get("entities", []):
            
            
            #Original name
            original_name = node.get("name", "Unknown")
            
            # Force UPPERCASE so "Logic" and "LOGIC" match
            name = node.get("name", "Unknown").strip().upper()
            node["name"] = name
            valid_nodes.add(name)
            normalized["entities"].append(node)
            
            if i < 3:
                log_debug("NORMALIZE", f"Entity: '{original_name}' -> '{name}'")

        # Process Relationships
        raw_edges = data.get("relationships", [])
        log_debug("CLEANER", f"Found {len(raw_edges)} raw relationships in JSON.")
        
        edges_kept = 0
        edges_dropped = 0
        
        for edge in raw_edges:
            src = edge.get("source", "Unknown").strip().upper()
            tgt = edge.get("target", "Unknown").strip().upper()
            
            # CHECK: Do endpoints exist?
            if src in valid_nodes and tgt in valid_nodes:
                edge["source"] = src
                edge["target"] = tgt
                normalized["relationships"].append(edge)
                edges_kept += 1
            else:
                edges_dropped += 1
                missing = []
                if src not in valid_nodes: missing.append(f"Source '{src}'")
                if tgt not in valid_nodes: missing.append(f"Target '{tgt}'")
                log_debug("DROP_EDGE", f"Dropping edge {src}->{tgt}. Missing: {', '.join(missing)}")

        log_debug("SUMMARY", f"Final Graph: {len(normalized['entities'])} Nodes, {len(normalized['relationships'])} Edges.")
        
        return normalized

    except json.JSONDecodeError as e:
        log_debug("ERROR", f"JSON Parse Failed: {e}")
        log_debug("ERROR_DATA", response) # Dump bad data to file
        return {"entities": [], "relationships": []}

# MODEL WRAPPER 
async def openai_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    # gets API key
    client = AsyncOpenAI()
    
    kwargs.pop("max_tokens", None)
    kwargs.pop("response_format", None)
    hashing_kv = kwargs.pop("hashing_kv", None)
    
    # Sanitize History (Convert Dicts to Strings for OpenAI)
    sanitized_history = []
    for msg in history_messages:
        content = msg.get("content")
        if not isinstance(content, str):
            
            log_debug("HISTORY", "Sanitizing a Dict in history to String.")
            
            
            # If it's a dict (from our cleaner), dump it back to string
            content = json.dumps(content)
        sanitized_history.append({"role": msg["role"], "content": content})

    messages = []
    if system_prompt: messages.append({"role": "system", "content": system_prompt})
    messages.extend(sanitized_history)
    messages.append({"role": "user", "content": prompt})
    
    # Check Input Size
    log_debug("INPUT", f"Sending prompt to OpenAI. Length: {len(prompt)} chars.")
    # Log the first 100 chars of the content to prove it's there
    content_preview = prompt.split("Content:")[-1][:100].replace("\n", " ")
    log_debug("INPUT_PREVIEW", f"Content sent to LLM: {content_preview}...")
    
    

    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        cached = await hashing_kv.get_by_id(args_hash)
        if cached: 
            log_debug("CACHE", "Hit! Returning cached response.")
            return cached["return"]

    try:
        response = await client.chat.completions.create(
            model=MODEL, messages=messages, **kwargs
        )
        raw_result = response.choices[0].message.content
        
        # --- PARSE HERE (FORCE IT) ---
        parsed_result = manual_clean_and_parse(raw_result)
        
        # Save to cache as DICT (so we skip parsing next time)
        if hashing_kv is not None:
            await hashing_kv.upsert({args_hash: {"return": parsed_result, "model": MODEL}})

        return parsed_result # Return Dict to library

    except Exception as e:
        log_debug("API_ERROR", str(e))
        return {"entities": [], "relationships": []}


    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": result, "model": MODEL}})

    return result
    
    # raw_response = response.choices[0].message.content
    # clean_json = raw_response.replace("```json", "").replace("```", "").strip()
    # data = json.loads(clean_json)
    # rag_string = dict_to_rag_string(data)

# --- 4. EXECUTION ---
if __name__ == "__main__":
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)

    # PASS THE CLEANER FUNCTION HERE
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        enable_llm_cache=True,
        best_model_func=openai_model_func,
        cheap_model_func=openai_model_func,
    )

    if os.path.exists(TEST_FILE):
        print(f"📂 Processing {os.path.basename(TEST_FILE)}...")
        with open(TEST_FILE, "r") as f:
            data = json.load(f)
            content = data.get("text", "") or "\n".join([c["text"] for c in data.get("chunks", [])])
            
            # Log the input size
            log_debug("MAIN", f"Loaded text length: {len(content)}")
            
            
            if len(content) > 15000: 
                
                log_debug("MAIN", "Trimmed text to 15000 chars.")
                content = content[:15000]

        rag.insert(content)
        
        # Verify
        graph_file = os.path.join(WORKING_DIR, "graph_chunk_entity_relation.graphml")
        if os.path.exists(graph_file):
            import networkx as nx
            G = nx.read_graphml(graph_file)
            print(f" Success! Graph Constructed with {G.number_of_nodes()} nodes.")
    else:
        print(" File not found.")
