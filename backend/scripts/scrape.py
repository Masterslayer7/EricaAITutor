import requests
from bs4 import BeautifulSoup
import os
import json
import re

# List of URLs to scrape
urls = [
    "https://pantelis.github.io/aiml-common/lectures/learning-problem/",
    "https://pantelis.github.io/aiml-common/lectures/regression/linear-regression/",
    "https://pantelis.github.io/aiml-common/lectures/empirical-risk/",
    "https://pantelis.github.io/aiml-common/lectures/optimization/sgd/",
    "https://pantelis.github.io/aiml-common/lectures/entropy/",
    "https://pantelis.github.io/aiml-common/lectures/optimization/maximum-likelihood/marginal_maximum_likelihood.html",
    "https://pantelis.github.io/aiml-common/lectures/optimization/maximum-likelihood/mle-gaussian-parameters.html",
    "https://pantelis.github.io/aiml-common/lectures/optimization/maximum-likelihood/conditional_maximum_likelihood.html",
    "https://pantelis.github.io/aiml-common/lectures/classification/classification-intro/",
    "https://pantelis.github.io/aiml-common/lectures/classification/logistic-regression/",
    "https://pantelis.github.io/aiml-common/lectures/optimization/whitening/",
    "https://pantelis.github.io/aiml-common/lectures/optimization/batch-normalization/",
    "https://pantelis.github.io/aiml-common/lectures/optimization/layer-normalization/",
    "https://pantelis.github.io/aiml-common/lectures/optimization/regularization/",
    "https://pantelis.github.io/aiml-common/lectures/dnn/dnn-intro/",
    "https://pantelis.github.io/aiml-common/lectures/dnn/backprop-intro/",
    "https://pantelis.github.io/aiml-common/lectures/dnn/backprop-dnn/",
    "https://pantelis.github.io/aiml-common/lectures/dnn/fashion-mnist-case-study.html",
    "https://pantelis.github.io/aiml-common/lectures/transfer-learning/transfer-learning-introduction.html",
    "https://pantelis.github.io/aiml-common/lectures/transfer-learning/transfer_learning_tutorial.html",
    "https://pantelis.github.io/aiml-common/lectures/cnn/cnn-intro/",
    "https://pantelis.github.io/aiml-common/lectures/cnn/cnn-layers/",
    "https://pantelis.github.io/aiml-common/lectures/cnn/cnn-example-architectures/",
    "https://pantelis.github.io/aiml-common/lectures/scene-understanding/feature-extraction-resnet/",
    "https://pantelis.github.io/aiml-common/lectures/scene-understanding/scene-understanding-intro/",
    "https://pantelis.github.io/aiml-common/lectures/scene-understanding/object-detection/detection-metrics/",
    "https://pantelis.github.io/aiml-common/lectures/scene-understanding/object-detection/rcnn/",
    "https://pantelis.github.io/aiml-common/lectures/scene-understanding/object-detection/fast-rcnn/",
    "https://pantelis.github.io/aiml-common/lectures/scene-understanding/object-detection/faster-rcnn/",
    "https://pantelis.github.io/aiml-common/lectures/scene-understanding/object-detection/yolo/introduction.html",
    "https://pantelis.github.io/aiml-common/lectures/scene-understanding/semantic-segmentation/maskrcnn/",
    "https://pantelis.github.io/aiml-common/lectures/rse/hmm-localization/",
    "https://pantelis.github.io/aiml-common/lectures/rse/recursive-state-estimation/",
    "https://pantelis.github.io/aiml-common/lectures/rse/discrete-bayesian-filter/discrete-bayesian-filter.html",
    "https://pantelis.github.io/aiml-common/lectures/rse/kalman-filters/one-dimensional-kalman-filters.html",
    "https://pantelis.github.io/aiml-common/lectures/rse/occupancy-mapping/",
    "https://pantelis.github.io/aiml-common/lectures/rse/slam/",
    "https://pantelis.github.io/aiml-common/lectures/nlp/nlp-introduction/nlp-pipelines/",
    "https://pantelis.github.io/aiml-common/lectures/nlp/nlp-introduction/tokenization/",
    "https://pantelis.github.io/aiml-common/lectures/nlp/nlp-introduction/word2vec/",
    "https://pantelis.github.io/aiml-common/lectures/nlp/nlp-introduction/word2vec/word2vec_from_scratch.html",
    "https://pantelis.github.io/aiml-common/lectures/nlp/nlp-introduction/word2vec/word2vec_tensorflow_tutorial.html",
    "https://pantelis.github.io/aiml-common/lectures/rnn/introduction/",
    "https://pantelis.github.io/aiml-common/lectures/rnn/simple-rnn/",
    "https://pantelis.github.io/aiml-common/lectures/rnn/lstm/",
    "https://pantelis.github.io/aiml-common/lectures/nlp/language-models/",
    "https://pantelis.github.io/aiml-common/lectures/nlp/language-models/rnn-language-model/",
    "https://pantelis.github.io/aiml-common/lectures/nlp/transformers/transformers-intro.html",
    "https://pantelis.github.io/aiml-common/lectures/nlp/transformers/singlehead-self-attention.html",
    "https://pantelis.github.io/aiml-common/lectures/nlp/transformers/multihead-self-attention.html",
    "https://pantelis.github.io/aiml-common/lectures/nlp/transformers/mlp.html",
    "https://pantelis.github.io/aiml-common/lectures/nlp/transformers/positional_embeddings.html",
    "https://pantelis.github.io/aiml-common/lectures/logical-reasoning/automated-reasoning/",
    "https://pantelis.github.io/aiml-common/lectures/logical-reasoning/propositional-logic/",
    "https://pantelis.github.io/aiml-common/lectures/logical-reasoning/logical-inference/",
    "https://pantelis.github.io/aiml-common/lectures/logical-reasoning/logical-agents/",
    "https://pantelis.github.io/aiml-common/lectures/planning/task-planning/",
    "https://pantelis.github.io/aiml-common/lectures/planning/task-planning/pddl/",
    "https://pantelis.github.io/aiml-common/lectures/mdp/",
    "https://pantelis.github.io/aiml-common/lectures/mdp/mdp-intro/mdp_intro.html",
    "https://pantelis.github.io/aiml-common/lectures/mdp/bellman-optimality-backup/",
    "https://pantelis.github.io/aiml-common/lectures/mdp/policy-improvement/",
    "https://pantelis.github.io/aiml-common/lectures/mdp/dynamic-programming-algorithms/policy-iteration/",
    "https://pantelis.github.io/aiml-common/lectures/mdp/dynamic-programming-algorithms/value-iteration/",
    "https://pantelis.github.io/aiml-common/lectures/reinforcement-learning/",
    "https://pantelis.github.io/aiml-common/lectures/reinforcement-learning/model-free-control/generalized-policy-iteration/",
    "https://pantelis.github.io/aiml-common/lectures/reinforcement-learning/prediction/monte-carlo.html",
    "https://pantelis.github.io/aiml-common/lectures/reinforcement-learning/prediction/temporal-difference.html",
    "https://pantelis.github.io/aiml-common/lectures/reinforcement-learning/model-free-control/greedy-monte-carlo/",
    "https://pantelis.github.io/aiml-common/lectures/reinforcement-learning/model-free-control/sarsa/"
]


output_dir = "scraped_json"
os.makedirs(output_dir, exist_ok=True)

# ---------- Helper functions ----------
def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n+", "\n", text).strip()
    return text

def chunk_text(text: str, chunk_size=800, overlap=200):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# ---------- Scrape and save ----------
for url in urls:
    try:
        print(f"Scraping: {url}")
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        raw_html = r.text

        text = clean_html(raw_html)
        chunks = chunk_text(text)

        # Save as JSON
        filename = url.rstrip("/").split("/")[-1]
        if filename == "":
            filename = "index"
        filepath = os.path.join(output_dir, f"{filename}.json")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "url": url,
                "text": text,
                "chunks": chunks
            }, f, ensure_ascii=False, indent=2)

        print(f"Saved → {filepath}, {len(chunks)} chunks")

    except Exception as e:
        print(f"Error scraping {url}: {e}")
