# Understanding Nano-Graphrag

```py
best_model_func = gpt_4o_mini_complet
```

## Desiered prompt according to documentation
```
("entity"<|>"Cruz"<|>"person"<|>"Cruz is associated with a vision of control and order, influencing the dynamics among other characters.")
```

# Querying the graph File: userinput.py

## The Entry Point: "Where are we?" (find_concept_node)

Before the system can explain anything, it has to find the specific "dot" (Node) in the graph that corresponds to the user's question.

* How it works: It takes the user's query (e.g., "Explain Automated Reasoning") and looks at every single node in your graph.
* The Logic: It performs a simple keyword match (if node in query).
* Why: In a Knowledge Graph, nodes are usually exact terms like "AUTOMATED REASONING" or "LOGISTIC REGRESSION." If the user mentions that term, we lock onto that node as our Target Node.

## The Traversal: "Gathering the Lesson Plan" (get_pedagogical_subgraph)

Once we have the Target Node, we don't just grab random neighbors. We look at the Edges  and classify them into three specific buckets.

We look at the neighbors of the Target Node and ask: "What is our relationship?"

### Bucket A: Prerequisites (Scaffolding)

* The Logic: The code checks edge descriptions for the word "prereq".
* The Graph Move: It looks for nodes that have a prereq_of relationship pointing to or from the Target.
* Pedagogical Goal: If the user asks about "ELBO," the system sees that "Jensen's Inequality" is a prerequisite. It grabs that node so it can explain it first. This is "Scaffolding"—building a foundation before adding the roof.

### Bucket B: Near Transfer (Siblings)

* The Logic: It checks edges for words like "transfer", "similar", or "contrast".
* The Graph Move: It looks for nodes that are "siblings" or "cousins" in the concept hierarchy.
* Pedagogical Goal: To test if a student truly understands a concept, you check if they can apply it to a similar but slightly different situation.

### Bucket C: Evidence (Resources & Examples)

* The Logic: It checks the entity_type of the neighbor.
* The Graph Move: It grabs any connected node labeled "resource" (Videos/Slides) or "example".
* Pedagogical Goal: A tutor needs concrete proof. This step pulls the URLs and specific "Security Robot Scenarios" you extracted earlier.

## Context Construction: "Writing the Cheat Sheet" (format_context)

Now that the script has gathered all these nodes, it creates a structured text block (a "Prompt") to send to GPT-4o. It organizes the information strictly:

    TOP: Prerequisite definitions (The "Review").

    MIDDLE: The Main Target Concept (The "Lesson").

    BOTTOM: Resources and Examples (The "Proof").

Why order matters: LLMs read top-to-bottom. By putting prerequisites first, we prime the model to explain the background before diving into the complex topic.


## Generation: "The Tutor Speaks" (generate_tutor_response)

Finally, we send that organized "Cheat Sheet" to GPT-4o with very specific instructions:

    "Start by briefly reviewing the PREREQUISITE CONCEPTS... Then, explain the TARGET CONCEPT... Use the provided RESOURCES..."

Because we did the hard work of finding the right nodes in Step 2, GPT-4o doesn't have to hallucinate. It just reads the context we gave it and synthesizes a perfect answer.

## 