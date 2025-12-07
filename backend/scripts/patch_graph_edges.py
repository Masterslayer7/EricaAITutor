import networkx as nx
import json
import os
import asyncio
from tqdm.asyncio import tqdm_asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError, APIError

# --- CONFIGURATION ---
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
INPUT_GRAPH_PATH = "/home/yugp/projects/EricaAITutor/backend/data/graph_edge_rework.graphml"
OUTPUT_GRAPH_PATH = "knowledge_graph_classified.graphml"
BACKUP_FILE_PATH = "classifications_backup.json" # New backup file
MODEL_NAME = "gpt-4o-mini"
# RATE LIMIT CONFIG
# Target: ~400 RPM (Safety Buffer for 500 RPM limit)
LAUNCH_DELAY = 0.15 
MAX_CONCURRENT_REQUESTS = 50 

def find_description_key(graph):
    """
    Intelligently finds the key used for edge descriptions (e.g., 'description' or 'd5').
    """
    if not graph.edges():
        return None
    
    # Get the data of the first edge
    first_edge_data = list(graph.edges(data=True))[0][2]
    
    # 1. Check for standard name
    if "description" in first_edge_data:
        return "description"
    
    # 2. Heuristic: Find the key with the longest string value (that isn't a chunk ID)
    best_key = None
    max_len = 0
    
    for key, value in first_edge_data.items():
        if isinstance(value, str):
            # Ignore chunk IDs or short labels
            if len(value) > max_len and "chunk-" not in value:
                max_len = len(value)
                best_key = key
                
    print(f"Detected description key: '{best_key}'")
    return best_key

def clean_node_id(node_id):
    """Removes extra quotes from GraphML IDs like '"LOGISTIC REGRESSION"'"""
    return str(node_id).replace('"', '').strip()

async def classify_edge_async(client, source, target, description, semaphore):
    async with semaphore:
        system_prompt = (
            "You are an expert Curriculum Designer. "
            "Classify the relationship between two concepts into EXACTLY ONE category:\n"
            "1. PREREQUISITE (Source is fundamental/required for Target)\n"
            "2. COMPONENT (Target is a part, step, or specific type of Source)\n"
            "3. ANALOGY (Concepts are peers, similar, or contrasting)\n"
            "4. EVIDENCE (Target is a specific example, resource, or file chunk)\n\n"
            "Respond with valid JSON only: {\"classification\": \"CATEGORY\"}"
        )
        
        user_message = f"Source: \"{source}\"\nTarget: \"{target}\"\nDescription: \"{description}\""

        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.0,
                    max_tokens=15,
                    response_format={"type": "json_object"}
                )
                
                content_str = response.choices[0].message.content
                content = json.loads(content_str)
                return content.get("classification", "ANALOGY")
                
            except RateLimitError:
                await asyncio.sleep(5 * (attempt + 1))
            except Exception as e:
                await asyncio.sleep(1)
        
        return "ANALOGY" 

async def process_graph():
    if not API_KEY:
        print("ERROR: OPENAI_API_KEY not found.")
        return

    client = AsyncOpenAI(api_key=API_KEY)
    
    print(f"--- Loading Graph from {INPUT_GRAPH_PATH} ---")
    try:
        G = nx.read_graphml(INPUT_GRAPH_PATH)
    except Exception as e:
        print(f"Error loading graph: {e}")
        return

    # --- STEP 1: DETECT DATA KEYS ---
    desc_key = find_description_key(G)
    if not desc_key:
        print("ERROR: Could not find a description field in the edge data.")
        return

    # --- STEP 2: PREPARE WORK ---
    edges_to_process = []
    edges_modified = 0

    print("Analyzing edges...")
    
    is_multigraph = G.is_multigraph()
    if is_multigraph:
        iterator = G.edges(keys=True, data=True)
    else:
        iterator = G.edges(data=True)
        
        
    for item in iterator:
        # Unpack based on graph type
        if is_multigraph:
            u, v, key, data = item
        else:
            u, v, data = item
            key = None # No key for simple graphs

        # Skip if done
        if "relationship_type" in data and data["relationship_type"] in ["PREREQUISITE", "COMPONENT", "ANALOGY", "EVIDENCE"]:
            continue
        
        description = data.get(desc_key, "")
        clean_u = clean_node_id(u)
        clean_v = clean_node_id(v)

        # Heuristic for Chunks
        if "chunk" in clean_u.lower() or "chunk" in clean_v.lower() or "chunk-" in description:
            # Update immediately
            if is_multigraph:
                G[u][v][key]["relationship_type"] = "EVIDENCE"
            else:
                G[u][v]["relationship_type"] = "EVIDENCE"
            edges_modified += 1
            continue
            
        edges_to_process.append((u, v, key, clean_u, clean_v, description))

    print(f"Total Edges: {len(G.edges())} | To Classify: {len(edges_to_process)}")

    # --- STEP 3: ASYNC EXECUTION ---
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = []

    async def rate_limited_task(u, v, key, c_u, c_v, desc, delay_steps):
        await asyncio.sleep(LAUNCH_DELAY * delay_steps)
        classification = await classify_edge_async(client, c_u, c_v, desc, semaphore)
        return (u, v, key, classification)

    print(f"Starting classification...")
    for index, (u, v, key, c_u, c_v, desc) in enumerate(edges_to_process):
        task = asyncio.create_task(rate_limited_task(u, v, key, c_u, c_v, desc, index))
        tasks.append(task)

    results = await tqdm_asyncio.gather(*tasks, desc="Classifying AI Edges")

    # --- SAVE BACKUP ---
    # We save the raw results list first. If the graph update crashes, you still have this file.
    print(f"\nSaving backup to {BACKUP_FILE_PATH}...")
    with open(BACKUP_FILE_PATH, "w") as f:
        # Convert tuple keys to list for JSON serialization if needed, or just dump structure
        # Structure: [(u, v, key, classification), ...]
        json.dump(results, f)
        
    # --- STEP 4: UPDATE & SAVE ---
    print("Updating Graph Data...")
    for u, v, key, classification in results:
        # Robust Update Logic
        if is_multigraph and key is not None:
            G[u][v][key]["relationship_type"] = classification
        else:
            # Simple Graph OR MultiGraph where key didn't matter (unlikely but safe)
            G[u][v]["relationship_type"] = classification
            
        edges_modified += 1

    print(f"--- Finished! Processed {edges_modified} edges. ---")
    nx.write_graphml(G, OUTPUT_GRAPH_PATH)
    print(f"Graph saved to: {OUTPUT_GRAPH_PATH}")

if __name__ == "__main__":
    asyncio.run(process_graph())