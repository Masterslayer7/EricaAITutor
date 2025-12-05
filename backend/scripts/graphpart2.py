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
WORKING_DIR = "./erica_graph_storage"
TEST_FILE = "/home/yugp/projects/EricaAITutor/backend/scripts/scraped_json/automated-reasoning.json"
DEBUG_FILE = "debug_trace.log"


logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

# PROMPT (Standard JSON) 
# We ask for the exact JSON structure nano-graphrag looks for keys in.
ERICA_JSON_PROMPT = """
-Goal-
You are an expert educational content analyzer for a tutoring AI named ERICA. Given a text document containing educational material (textbook excerpts, lecture notes, or transcripts) and a list of entity types, your goal is to extract a knowledge graph that helps students understand concepts and their relationships.

-Steps-
1. Identify all educational entities. For each identified entity, extract the following information:
- entity_name: Name of the concept, term, or object, capitalized.
- entity_type: One of the following types: [{entity_types}]
- entity_description: A comprehensive, pedagogical description. Focus on what the concept IS, how it functions, and its role in the broader subject.
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *pedagogically related*.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation of the educational link. Does one concept prerequisite the other? Is one an example of the other? Does one contrast with the other?
- relationship_strength: a numeric score (1-10) indicating how tight the connection is. (e.g., A strict prerequisite is a 9 or 10; a loose thematic link is a 3 or 4).
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 1:

Entity_types: [concept, definition, prerequisite, misconception, example]
Text:
To fully understand the Chain Rule in calculus, a student must first have a solid grasp of composite functions. The Chain Rule states that the derivative of a composite function is the derivative of the outer function evaluated at the inner function times the derivative of the inner function. A common error students make is forgetting to multiply by the derivative of the inner function—this is often called "dropping the inner derivative."

Output:
("entity"{tuple_delimiter}"Chain Rule"{tuple_delimiter}"concept"{tuple_delimiter}"The Chain Rule is a formula to compute the derivative of a composite function."){record_delimiter}
("entity"{tuple_delimiter}"Composite Functions"{tuple_delimiter}"prerequisite"{tuple_delimiter}"Composite functions are functions composed of other functions, serving as the foundational structure upon which the Chain Rule operates."){record_delimiter}
("entity"{tuple_delimiter}"Dropping The Inner Derivative"{tuple_delimiter}"misconception"{tuple_delimiter}"A common calculation error where the student derives the outer function but forgets to multiply by the derivative of the inner term."){record_delimiter}
("relationship"{tuple_delimiter}"Composite Functions"{tuple_delimiter}"Chain Rule"{tuple_delimiter}"Understanding Composite Functions is a strict prerequisite for applying the Chain Rule."{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Chain Rule"{tuple_delimiter}"Dropping The Inner Derivative"{tuple_delimiter}"Dropping the inner derivative is a specific failure mode associated with applying the Chain Rule."{tuple_delimiter}8){completion_delimiter}
#############################
Example 2:

Entity_types: [concept, code_structure, library, definition]
Text:
In Python, a List Comprehension provides a concise way to create lists. It consists of brackets containing an expression followed by a 'for' clause, then zero or more 'for' or 'if' clauses. This is often more readable and faster than using a standard 'for loop' to append items to a list. However, if the logic is too complex, readability suffers, and a standard loop is preferred.

Output:
("entity"{tuple_delimiter}"List Comprehension"{tuple_delimiter}"code_structure"{tuple_delimiter}"A syntactic construct in Python that allows for the creation of new lists based on existing iterables in a concise, readable single line."){record_delimiter}
("entity"{tuple_delimiter}"For Loop"{tuple_delimiter}"code_structure"{tuple_delimiter}"A standard control flow statement for iterating over a sequence, which serves as the verbose alternative to list comprehensions."){record_delimiter}
("entity"{tuple_delimiter}"Readability"{tuple_delimiter}"concept"{tuple_delimiter}"A measure of how easily code can be understood by a human, which creates a trade-off between list comprehensions and loops."){record_delimiter}
("relationship"{tuple_delimiter}"List Comprehension"{tuple_delimiter}"For Loop"{tuple_delimiter}"List Comprehension is an alternative implementation to the For Loop, often cited as being faster and more concise."{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"List Comprehension"{tuple_delimiter}"Readability"{tuple_delimiter}"List Comprehensions generally improve readability, though excessive complexity can degrade it."{tuple_delimiter}6){completion_delimiter}
#############################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
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



# def manual_clean_and_parse(response: str) -> dict:
#     """
#     1. Strips markdown.
#     2. Parses Tuple-Delimited Strings (Microsoft GraphRAG style).
#        Expected Format: ("entity"<|>"Name"<|>"Type"<|>"Desc")
#     3. Normalizes all Names to UPPERCASE.
#     """
#     log_debug("CLEANER", "Received raw string from LLM. Length: " + str(len(response)))
    
#     # --- 1. STRING CLEANING ---
#     # Remove markdown code blocks if the LLM was polite enough to add them
#     clean_text = response.replace("```text", "").replace("```", "").strip()
    
#     # Define our delimiters (MUST MATCH YOUR PROMPT CONFIGURATION)
#     # Assuming you used <|> in the prompt injection
#     TUPLE_DELIMITER = "<|>" 
    
#     normalized = {"entities": [], "relationships": []}
#     valid_nodes = set()

#     try:
#         # Split into lines (handling standard newlines or custom record delimiters)
#         # If your prompt uses {record_delimiter}, split by that. 
#         # Here we assume standard newlines for simplicity.
#         lines = clean_text.split('\n')

#         # --- 2. PARSE ENTITIES ---
#         for line in lines:
#             line = line.strip()
#             # Look for lines starting with ("entity"
#             if line.startswith('("entity"'):
#                 # Remove wrapping (" and ")
#                 content = line[2:-2] 
#                 parts = content.split(TUPLE_DELIMITER)
                
#                 # Cleanup quotes around the values: "Name" -> Name
#                 # We expect 4 parts: "entity", "Name", "Type", "Desc"
#                 if len(parts) >= 4:
#                     entity_name = parts[1].strip('"').upper() # NORMALIZE HERE
#                     entity_type = parts[2].strip('"')
#                     entity_desc = parts[3].strip('"')
                    
#                     # Store
#                     normalized["entities"].append({
#                         "name": entity_name,
#                         "type": entity_type,
#                         "description": entity_desc
#                     })
#                     valid_nodes.add(entity_name)

#         log_debug("CLEANER", f"Extracted {len(valid_nodes)} valid entities.")

#         # --- 3. PARSE RELATIONSHIPS ---
#         edges_kept = 0
#         edges_dropped = 0
        
#         for line in lines:
#             line = line.strip()
#             # Look for lines starting with ("relationship"
#             if line.startswith('("relationship"'):
#                 content = line[2:-2]
#                 parts = content.split(TUPLE_DELIMITER)
                
#                 # We expect 5 parts: "relationship", "Src", "Tgt", "Desc", "Score"
#                 if len(parts) >= 5:
#                     src = parts[1].strip('"').upper() # NORMALIZE HERE
#                     tgt = parts[2].strip('"').upper() # NORMALIZE HERE
#                     desc = parts[3].strip('"')
#                     try:
#                         score = int(parts[4].strip('"'))
#                     except:
#                         score = 5 # Default if parsing fails

#                     # CHECK: Do endpoints exist?
#                     if src in valid_nodes and tgt in valid_nodes:
#                         normalized["relationships"].append({
#                             "source": src,
#                             "target": tgt,
#                             "description": desc,
#                             "weight": score
#                         })
#                         edges_kept += 1
#                     else:
#                         edges_dropped += 1
#                         missing = []
#                         if src not in valid_nodes: missing.append(f"Source '{src}'")
#                         if tgt not in valid_nodes: missing.append(f"Target '{tgt}'")
#                         # log_debug("DROP_EDGE", f"Dropping {src}->{tgt}. Missing: {missing}")

#         log_debug("SUMMARY", f"Final Graph: {len(normalized['entities'])} Nodes, {len(normalized['relationships'])} Edges. Dropped {edges_dropped} orphan edges.")
        
#         return normalized

#     except Exception as e:
#         log_debug("ERROR", f"Parsing Failed: {e}")
#         log_debug("ERROR_DATA", response[:500]) # Dump start of bad data
#         return {"entities": [], "relationships": []}


# # MODEL WRAPPER 
# async def openai_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
#     # gets API key
#     client = AsyncOpenAI()
    
#     kwargs.pop("max_tokens", None)
#     kwargs.pop("response_format", None)
#     hashing_kv = kwargs.pop("hashing_kv", None)
    
#     # Sanitize History (Convert Dicts to Strings for OpenAI)
#     sanitized_history = []
#     for msg in history_messages:
#         content = msg.get("content")
#         if not isinstance(content, str):
            
#             log_debug("HISTORY", "Sanitizing a Dict in history to String.")
            
            
#             # If it's a dict (from our cleaner), dump it back to string
#             content = json.dumps(content)
#         sanitized_history.append({"role": msg["role"], "content": content})

#     messages = []
#     if system_prompt: messages.append({"role": "system", "content": system_prompt})
#     messages.extend(sanitized_history)
#     messages.append({"role": "user", "content": prompt})
    
#     # Check Input Size
#     log_debug("INPUT", f"Sending prompt to OpenAI. Length: {len(prompt)} chars.")
#     # Log the first 100 chars of the content to prove it's there
#     content_preview = prompt.split("Content:")[-1][:100].replace("\n", " ")
#     log_debug("INPUT_PREVIEW", f"Content sent to LLM: {content_preview}...")
    
    

#     if hashing_kv is not None:
#         args_hash = compute_args_hash(MODEL, messages)
#         cached = await hashing_kv.get_by_id(args_hash)
#         if cached: 
#             log_debug("CACHE", "Hit! Returning cached response.")
#             return cached["return"]

#     try:
#         response = await client.chat.completions.create(
#             model=MODEL, messages=messages, **kwargs
#         )
#         raw_result = response.choices[0].message.content
        
#         # --- PARSE HERE (FORCE IT) ---
#         parsed_result = manual_clean_and_parse(raw_result)
        
#         # Save to cache as DICT (so we skip parsing next time)
#         if hashing_kv is not None:
#             await hashing_kv.upsert({args_hash: {"return": parsed_result, "model": MODEL}})

#         return parsed_result # Return Dict to library

#     except Exception as e:
#         log_debug("API_ERROR", str(e))
#         return {"entities": [], "relationships": []}
    
#     # raw_response = response.choices[0].message.content
#     # clean_json = raw_response.replace("```json", "").replace("```", "").strip()
#     # data = json.loads(clean_json)
#     # rag_string = dict_to_rag_string(data)

# # --- 4. EXECUTION ---
# if __name__ == "__main__":
#     if os.path.exists(WORKING_DIR):
#         shutil.rmtree(WORKING_DIR)

#     # PASS THE CLEANER FUNCTION HERE
#     rag = GraphRAG(
#         working_dir=WORKING_DIR,
#         enable_llm_cache=True,
#         best_model_func=openai_model_func,
#         cheap_model_func=openai_model_func,
#     )

#     if os.path.exists(TEST_FILE):
#         print(f"📂 Processing {os.path.basename(TEST_FILE)}...")
#         with open(TEST_FILE, "r") as f:
#             data = json.load(f)
#             content = data.get("text", "") or "\n".join([c["text"] for c in data.get("chunks", [])])
            
#             # Log the input size
#             log_debug("MAIN", f"Loaded text length: {len(content)}")
            
            
#             if len(content) > 15000: 
                
#                 log_debug("MAIN", "Trimmed text to 15000 chars.")
#                 content = content[:15000]

#         rag.insert(content)
        
#         # Verify
#         graph_file = os.path.join(WORKING_DIR, "graph_chunk_entity_relation.graphml")
#         if os.path.exists(graph_file):
#             import networkx as nx
#             G = nx.read_graphml(graph_file)
#             print(f" Success! Graph Constructed with {G.number_of_nodes()} nodes.")
#     else:
#         print(" File not found.")

async def openai_raw_debug_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    client = AsyncOpenAI() 
    
    # Remove arguments that might cause errors if passed blindly
    kwargs.pop("max_tokens", None)
    kwargs.pop("response_format", None)
    kwargs.pop("hashing_kv", None)

    # Basic Message Construction
    messages = []
    if system_prompt: 
        messages.append({"role": "system", "content": system_prompt})
    # Add history if strictly necessary, usually empty for GraphRAG extraction
    messages.extend(history_messages) 
    messages.append({"role": "user", "content": prompt})
    
    print(f"\n🚀 Sending Prompt to {MODEL}...")

    try:
        response = await client.chat.completions.create(
            model=MODEL, 
            messages=messages, 
            **kwargs
        )
        
        raw_result = response.choices[0].message.content

        # --- 🛑 HERE IS THE DEBUG PRINTOUT 🛑 ---
        print("\n" + "!"*50)
        print(" RAW LLM RESPONSE START ")
        print("!"*50)
        print(raw_result)
        print("!"*50)
        print(" RAW LLM RESPONSE END ")
        print("!"*50 + "\n")
        
        # We return the raw string. 
        # NOTE: The GraphRAG library might crash here if it expects a Dict/List.
        # That is okay; we just wanted to see the printout above.
        return raw_result 

    except Exception as e:
        print(f"❌ API ERROR: {str(e)}")
        return ""

# --- 2. MAIN EXECUTION ---
if __name__ == "__main__":
    # Clear previous runs to ensure we actually trigger the LLM
    if os.path.exists(WORKING_DIR):
        print(f"🧹 Clearing {WORKING_DIR}...")
        shutil.rmtree(WORKING_DIR)

    # Initialize GraphRAG with our RAW DEBUGGER
    # We disable cache to ensure it hits the API and prints to console
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        enable_llm_cache=False, 
        best_model_func=openai_raw_debug_func, 
        cheap_model_func=openai_raw_debug_func, 
    )

    if os.path.exists(TEST_FILE):
        print(f"📂 Loading {TEST_FILE}...")
        with open(TEST_FILE, "r") as f:
            data = json.load(f)
            # Handle standard JSON or chunked JSON
            content = data.get("text", "") or "\n".join([c["text"] for c in data.get("chunks", [])])
            
            # Limit content size for this debug test to save money/time
            if len(content) > 5000:
                print("✂️  Trimming text to first 5000 chars for debug test.")
                content = content[:5000]

        print("🏃 Starting Insert...")
        
        # This will trigger the openai_raw_debug_func
        try:
            rag.insert(content)
        except Exception as e:
            print(f"\n⚠️  Pipeline stopped (Expected): {e}")
            print("Check the console output above to see the raw LLM response.")
            
    else:
        print(f"❌ File {TEST_FILE} not found. Please provide a valid JSON input file.")