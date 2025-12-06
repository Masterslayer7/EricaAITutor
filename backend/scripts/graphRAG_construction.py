import asyncio
import os
import json
import logging
import glob
from dotenv import load_dotenv
from openai import AsyncOpenAI
from nano_graphrag import GraphRAG
from nano_graphrag.prompt import PROMPTS

# CONFIGURATION 
load_dotenv()
MODEL = "gpt-4o-mini"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT = os.path.dirname(SCRIPT_DIR)
WORKING_DIR = os.path.join(BACKEND_ROOT, "data", "erica_graph_storage")
INPUT_DIR = os.path.join(BACKEND_ROOT, "data", "scraped_json")


logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

# PROMPT inspired from the prompt.py file from nano-GraphRAG library
# We ask for the exact tuple structure nano-graphrag looks for when creating a graph structure for networkx.
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

# Function that calles openAI api with dynamic env api key
async def openai_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    
    # gets API key
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
    
    if len(history_messages) > 10:
        print(f"⚠️ WARNING: History is getting huge! ({len(history_messages)} messages)")
    
    # DEBUG: Check total estimated characters
    total_chars = sum([len(m['content']) for m in messages if m['content']])
    print(f" Payload size: {total_chars} chars")
    
    max_retries = 10
    base_delay = 5 # Start waiting 5 seconds
    
    print(f"\n Sending Prompt to {MODEL}...")

    for attempt in range(max_retries):
        #Send api call
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=messages, 
                **kwargs
            )
            
            return response.choices[0].message.content

        except Exception as e:
            error_msg = str(e)
            
            # CHECK FOR RATE LIMIT (429)
            if "429" in error_msg:
                
                # Calculate wait time: 5s, 10s, 20s, 40s... + random jitter
                wait_time = (base_delay * (2 ** attempt)) + (random.random() * 3)
                print(f" Rate Limit (429) hit. Pausing for {wait_time:.1f}s before retry {attempt+1}/{max_retries}...")
                
                # IMPORTANT: Async sleep so we don't block the whole script
                await asyncio.sleep(wait_time) 
            
            else:
                # If it's a real error (like 400 Bad Request), print and fail.
                print(f"❌ API ERROR: {error_msg}")
                return ""
    
    print("❌ Failed after max retries. Skipping chunk.")
    return ""

if __name__ == "__main__":

    # Initialize GraphRAG with our RAW DEBUGGER
    # We disable cache to ensure it hits the API and prints to console
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        enable_llm_cache=True, 
        best_model_func=openai_func, 
        cheap_model_func=openai_func, 
        cheap_model_max_async=3, 
        best_model_max_async=3
    )
    
    json_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    print(f"found {len(json_files)} JSON files to process.")
    
    for i, file_path in enumerate(json_files):
        
        # get file name
        filename = os.path.basename(file_path)
        
        print(f" Processing {i+1}/{len(json_files)}: {os.path.basename(file_path)}...")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                # Load the json into data
                data = json.load(f)
                
                # Get the full text of the page
                full_text = data.get("text", "") 

                # The main call to insert it into the rag model to run through whole pipeline
                rag.insert(full_text)
                print(f" Inserted ({len(full_text)} chars)")

        except Exception as e:
            print(f"\n Error processing {filename}: {e}")

    print("\n __________________ Build Complete __________________________")