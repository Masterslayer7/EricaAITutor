import os
import networkx as nx
import logging
from dotenv import load_dotenv
from openai import OpenAI 
from flask import Flask, request, jsonify
from flask_cors import CORS
from markdown import markdown




app = Flask(__name__)
CORS(app)

# CONFIGURATION 
load_dotenv()
client = OpenAI() # Uses OPENAI_API_KEY from .env

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT = os.path.dirname(SCRIPT_DIR)
WORKING_DIR = os.path.join(BACKEND_ROOT, "data", "erica_graph_storage")
GRAPH_PATH = os.path.join(BACKEND_ROOT, "data", "knowledge_graph_classified.graphml")

# NODE MAPPING (Query -> Entry Point) 
def find_concept_node(graph, query):
    """
    Maps a user query to a specific node in the graph using keyword matching.
    Rationale: In a specialized educational graph, node names (Concepts) 
    are usually distinct technical terms. Keyword matching is precise and low-latency.
    """
    
    print(" -- Finding concept node -- ")
    query_upper = query.upper() # Makes query capital
    best_match = None
    best_score = 0

    for node in graph.nodes():
        
        # Gets rid of the quotes around the node
        node_clean = str(node).upper().replace('"', '').replace("'", "").strip()
        
        # Simple Jaccard/Overlap matching
        if node_clean in query_upper or query_upper in node_clean:
            score = len(node) # Prefer longer, more specific matches
            if score > best_score:
                best_score = score
                best_match = node
    
    print(f"Here is the best match Node: {best_match}")
    return best_match

#  SUBGRAPH SELECTION (The Core Logic) 
def get_pedagogical_subgraph(graph, target_node):
    """
    Selects a subgraph centered on the target node but explicitly 
    prioritizes educational edges.
    
    Rationale:
    - We traverse 'prereq_of' backwards to build a scaffolding chain.
    - We grab 'near_transfer' for breadth testing.
    - We strictly collect resources attached to these specific concepts.
    """
    context_nodes = {target_node}
    
    # Helper to handle MultiDiGraph vs DiGraph edge data
    def get_edge_data(u, v):
        if graph.is_multigraph():
            return graph[u][v].values() # List of edge dicts
        return [graph[u][v]] # List containing single edge dict
    
    def get_relationship_from_edge(attrs):
        # First, try the obvious key
        if "relationship_type" in attrs:
            return attrs["relationship_type"].upper()
        
        # Fallback: Scan ALL values in the edge dictionary
        # We look for our known keywords.
        valid_types = {"PREREQUISITE", "COMPONENT", "ANALOGY", "EVIDENCE"}
        for value in attrs.values():
            if isinstance(value, str) and value.upper() in valid_types:
                return value.upper()
        
        return "UNKNOWN"

    
    # Scaffolding (Find Prerequisites)
    # Walk backwards: Who is a prereq of the target?
    prereqs = []
    visited_parents = {target_node}
    stack = [target_node]
    found_prereqs_temp = []

    while stack:
        current = stack.pop()
        print(f"Current node in prereq search is {current}")

        # FIX: Use neighbors() because your graph is undirected
        try:
            for parent in graph.neighbors(current):
                
                if parent in visited_parents:
                    continue

                edges = get_edge_data(parent, current)
                is_valid_parent = False
                
                print(f"Checking {current} <-> {parent}") # Uncomment to see it working

                for attrs in edges:
                    rtype = get_relationship_from_edge(attrs)
                    print(rtype)
                    
                    # LOGIC:
                    # Even though the graph is undirected, we treat the relationship semantically.
                    # If the edge is "PREREQUISITE" or "COMPONENT", we accept it as a parent node.
                    if rtype in ["PREREQUISITE", "COMPONENT"]:
                        is_valid_parent = True
                        break
                
                if is_valid_parent:
                    print(f"  -> Found Prereq: {parent}")
                    visited_parents.add(parent)
                    found_prereqs_temp.append(parent)
                    context_nodes.add(parent)
                    stack.append(parent) 
                    
        except Exception as e:
             print(f"Error on node {current}: {e}") 
             continue
    
    # Reverse list so the most fundamental concept comes first (Root -> Leaf)
    prereqs = found_prereqs_temp[::-1]

    # B. Near Transfer (Siblings)
    # Check Outgoing Analogies (Target -> Sibling)
    siblings = []
    for neighbor in graph.neighbors(target_node):
        if neighbor in context_nodes: 
            continue
            
        for attrs in get_edge_data(target_node, neighbor):
            if get_relationship_from_edge(attrs) == "ANALOGY":
                siblings.append(neighbor)
                context_nodes.add(neighbor)
                break

    # C. Resources & Examples (Evidence)
    current_context = list(context_nodes) # Snapshot to avoid 'Set changed size' error
    evidence = []
    for node in current_context:
        try:
            # FIX: Use neighbors() instead of successors()
            for neighbor in graph.neighbors(node):
                
                # specific check to avoid cycles or duplicates
                if neighbor in context_nodes: 
                    continue
                
                # Get edge data for NODE <-> NEIGHBOR
                edges = get_edge_data(node, neighbor)
                is_evidence = False
                
                for attrs in edges:
                    # Use your robust helper function
                    rtype = get_relationship_from_edge(attrs)
                    
                    # 1. Strict Check: Is the relationship labeled EVIDENCE?
                    if rtype == "EVIDENCE":
                        is_evidence = True
                    
                    # 2. Heuristic Check: Does the node name look like a chunk?
                    # (Useful if the edge label is missing/wrong)
                    elif "chunk" in str(neighbor).lower():
                        is_evidence = True
                        
                    if is_evidence: 
                        break # Stop checking other edges if one confirms it
                        
                if is_evidence:
                    evidence.append(neighbor)
                    context_nodes.add(neighbor)
                    
        except Exception:
            continue
        
    print(context_nodes)
    print(prereqs)
    print(siblings)
    print(evidence)

    return context_nodes, prereqs, siblings, evidence

# CONTEXT BUILDER 
def format_context(graph, context_nodes, prereqs, target):
    """
    Formats the subgraph into a prompt, ordering from Simple -> Complex.
    """
    lines = []
    
    # Prerequisites (Scaffolding) first
    if prereqs:
        lines.append(f"--- PREREQUISITE CONCEPTS (Scaffolding for {target}) ---")
        for node in prereqs:
            desc = graph.nodes[node].get("description", "No definition")
            lines.append(f"Concept: {node}\nDetails: {desc}\n")

    # The Main Concept
    lines.append(f"--- TARGET CONCEPT: {target} ---")
    desc = graph.nodes[target].get("description", "No definition")
    lines.append(f"Definition: {desc}\n")

    # Resources/Examples
    lines.append("--- RESOURCES & EXAMPLES ---")
    for node in context_nodes:
        if node not in prereqs and node != target:
            data = graph.nodes[node]
            lines.append(f"Item: {node}\nInfo: {data.get('description', '')}")

    return "\n".join(lines)

# GENERATION 
def generate_tutor_response(query, context_str):
    prompt = f"""
    You are Erica, an expert AI Tutor.
    
    USER QUERY: "{query}"
    
    Use the Knowledge Graph context below to answer.
    
     PEDAGOGICAL INSTRUCTIONS:
    1. Start by briefly reviewing the PREREQUISITE CONCEPTS to scaffold the learning.
    2. Then, explain the TARGET CONCEPT in depth.
    3. Use the provided RESOURCES/EXAMPLES to illustrate.
    4. Finally, mention related concepts (Near Transfer) to broaden understanding.
    
    CONTEXT SUBGRAPH:
    {context_str}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# EXECUTION FLOW 
@app.route("/ask", methods=["POST"])
def user_input_flow():
    print(" Loading Knowledge Graph...")
    if not os.path.exists(GRAPH_PATH):
        print(f" Error: Graph not found at {GRAPH_PATH}. Run build script first.")
        exit()
        
    G = nx.read_graphml(GRAPH_PATH)
    print(f" Loaded {G.number_of_nodes()} concepts.")

    
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    user_query = data["question"]
    
    print(f"\n User Query: {user_query}")

    # Map to Node
    target_node = find_concept_node(G, user_query)
    
    if target_node:
        print(f" Mapped to Graph Node: {target_node}")
        
        # Select Subgraph
        all_nodes, prereqs, siblings, evidence = get_pedagogical_subgraph(G, target_node)
        print(f" Selected Subgraph: {len(all_nodes)} nodes")
        print(f"   - Prerequisites: {prereqs}")
        print(f"   - Siblings: {siblings}")
        print(f"   - Evidence: {len(evidence)} items")
        
        # Generate
        context_str = format_context(G, all_nodes, prereqs, target_node)
        answer = generate_tutor_response(user_query, context_str)
        
        print("\n Ericas's Answer:\n")
        print(answer)
        return markdown(answer), 200, {"Content-Type": "text/plain; charset=utf-8"}
    else:
        print(" Concept not found in Knowledge Graph.")
        return "Concept not found in Knowledge Graph.", 404, {"Content-Type": "text/plain; charset=utf-8"}
    
if __name__ == "__main__":
    app.run(debug=True)
