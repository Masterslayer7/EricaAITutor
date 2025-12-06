import os
import networkx as nx
import logging
from dotenv import load_dotenv
from openai import OpenAI 

# CONFIGURATION 
load_dotenv()
client = OpenAI() # Uses OPENAI_API_KEY from .env
GRAPH_PATH = "/home/yugp/projects/EricaAITutor/backend/data/erica_graph_storage/graph_chunk_entity_relation.graphml"

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
        
        # Simple Jaccard/Overlap matching
        if node in query_upper or query_upper in node:
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
    context_nodes = set([target_node])
    
    
    
    # Scaffolding (Find Prerequisites)
    # Walk backwards: Who is a prereq of the target?
    prereqs = []
    try:
        # Predecessors in a directed graph (if edges follow flow A->B)
        # Assuming prereq_of points Pre -> Post. 
        # We look at 'in_edges' or search manually if undirected.
        for neighbor in graph.neighbors(target_node):
            edge_data = graph.get_edge_data(target_node, neighbor)
            # Check edge descriptions stored by our builder
            # (NetworkX stores multigraph edges in a dict, we check values)
            for key, attrs in edge_data.items():
                desc = attrs.get("description", "").lower()
                if "prereq" in desc:
                    prereqs.append(neighbor)
                    context_nodes.add(neighbor)
    except Exception:
        pass # Handle graph variations

    # B. Near Transfer (Siblings)
    siblings = []
    for neighbor in graph.neighbors(target_node):
        if neighbor in context_nodes: continue
        edge_data = graph.get_edge_data(target_node, neighbor)
        for key, attrs in edge_data.items():
            desc = attrs.get("description", "").lower()
            if "transfer" in desc or "similar" in desc or "contrast" in desc:
                siblings.append(neighbor)
                context_nodes.add(neighbor)

    # C. Resources & Examples (Evidence)
    evidence = []
    for node in list(context_nodes):
        for neighbor in graph.neighbors(node):
            if neighbor in context_nodes: continue
            # Check if neighbor is a resource/example node (based on description/type)
            node_data = graph.nodes[neighbor]
            node_desc = node_data.get("description", "").lower()
            node_type = node_data.get("entity_type", "").lower()
            
            if "resource" in node_type or "example" in node_type or "url" in node_desc:
                evidence.append(neighbor)
                context_nodes.add(neighbor)

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
if __name__ == "__main__":
    print(" Loading Knowledge Graph...")
    if not os.path.exists(GRAPH_PATH):
        print(f" Error: Graph not found at {GRAPH_PATH}. Run build script first.")
        exit()
        
    G = nx.read_graphml(GRAPH_PATH)
    print(f" Loaded {G.number_of_nodes()} concepts.")

    # Hard coded Query
    user_query = "Explain Automated Reasoning" 
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
    else:
        print(" Concept not found in Knowledge Graph.")