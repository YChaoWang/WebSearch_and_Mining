import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import re

# Initialize stemmer and stop words
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))


def load_relevance_data(filepath):
    rel_data = {}
    with open(filepath, "r") as f:
        for line in f:
            query_id, doc_ids = line.strip().split("\t")
            # Extract just the numeric parts of the IDs
            query_id = re.search(r"\d+", query_id).group()
            doc_ids = eval(doc_ids)
            # Convert document IDs to just numbers
            doc_ids = [re.search(r"\d+", str(doc_id)).group() for doc_id in doc_ids]
            rel_data[query_id] = doc_ids

    rel_data_list = [
        (query_id, doc_id, 1)
        for query_id, doc_ids in rel_data.items()
        for doc_id in doc_ids
    ]
    return pd.DataFrame(rel_data_list, columns=["QueryID", "DocID", "Relevance"])


def load_documents(collection_path):
    documents = {}
    for filename in os.listdir(collection_path):
        # Extract just the numeric ID from filename
        doc_id = re.search(r"\d+", filename).group()
        with open(os.path.join(collection_path, filename), "r", encoding="utf-8") as f:
            documents[doc_id] = f.read().lower()  # Convert to lowercase
    return documents


def load_queries(queries_path):
    queries = {}
    for filename in os.listdir(queries_path):
        # Extract just the numeric ID from filename
        query_id = re.search(r"\d+", filename).group()
        with open(os.path.join(queries_path, filename), "r", encoding="utf-8") as f:
            queries[query_id] = f.read().lower()  # Convert to lowercase
    return queries


def preprocess_text(text):
    # Remove punctuation and convert to lowercase
    text = re.sub(r"[^\w\s]", "", text.lower())
    tokens = text.split()
    # Remove stopwords and apply stemming
    processed_tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    return " ".join(processed_tokens)


def compute_cosine_similarity(query, documents):
    # Create a list of all documents plus the query, maintaining the order
    doc_ids = list(documents.keys())
    texts = [documents[doc_id] for doc_id in doc_ids] + [query]

    # Compute TF-IDF and cosine similarity
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    cosine_sim = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])

    # Return similarities with their corresponding document IDs
    similarities = [(doc_id, sim) for doc_id, sim in zip(doc_ids, cosine_sim[0])]
    return similarities


def calculate_metrics_at_k(relevant_docs, ranked_docs, k=10):
    if not relevant_docs:
        return 0, 0, 0

    relevant_docs = set(relevant_docs)
    ranked_docs = ranked_docs[:k]

    # MRR@k
    mrr = 0
    for i, doc in enumerate(ranked_docs):
        if doc in relevant_docs:
            mrr = 1 / (i + 1)
            break

    # MAP@k
    hits = 0
    sum_precisions = 0
    for i, doc in enumerate(ranked_docs):
        if doc in relevant_docs:
            hits += 1
            sum_precisions += hits / (i + 1)
    map_score = sum_precisions / min(len(relevant_docs), k) if relevant_docs else 0

    # Recall@k
    hits = len(set(ranked_docs) & relevant_docs)
    recall = hits / len(relevant_docs) if relevant_docs else 0

    return mrr, map_score, recall


def evaluate_ir_system(queries_path, collections_path, rel_file):
    print("Loading data...")
    relevance_data = load_relevance_data(rel_file)
    documents = load_documents(collections_path)
    queries = load_queries(queries_path)

    # Preprocess all documents once
    print("Preprocessing documents...")
    docs_preprocessed = {
        doc_id: preprocess_text(text)
        for doc_id, text in tqdm(documents.items(), desc="Preprocessing")
    }

    results = defaultdict(list)
    print("Processing queries and calculating metrics...")

    for query_id, query_text in tqdm(queries.items(), desc="Evaluating Queries"):
        # Get relevant documents for this query
        relevant_docs = relevance_data[relevance_data["QueryID"] == query_id][
            "DocID"
        ].tolist()

        # Preprocess query
        query_preprocessed = preprocess_text(query_text)

        # Compute similarities and rank documents
        similarities = compute_cosine_similarity(query_preprocessed, docs_preprocessed)
        ranked_docs = [
            doc_id
            for doc_id, _ in sorted(similarities, key=lambda x: x[1], reverse=True)
        ]

        # Calculate metrics
        mrr, map_score, recall = calculate_metrics_at_k(relevant_docs, ranked_docs)

        results["MRR@10"].append(mrr)
        results["MAP@10"].append(map_score)
        results["Recall@10"].append(recall)

    # Print summary with more detail
    print("\nTask4: Evaluation IR System")
    print(f"Number of queries processed: {len(queries)}")
    print(f"Number of documents in collection: {len(documents)}")
    print(f"MRR@10: {np.mean(results['MRR@10']):.4f}")
    print(f"MAP@10: {np.mean(results['MAP@10']):.4f}")
    print(f"Recall@10: {np.mean(results['Recall@10']):.4f}")


# File paths
# base_path = "/Users/wangyichao/WebSearch_and_WebMining/smaller_dataset"
# queries_path = os.path.join(base_path, "queries")
# collections_path = os.path.join(base_path, "collections")
# rel_file = os.path.join(base_path, "rel.tsv")

# Run evaluation
# evaluate_ir_system(queries_path, collections_path, rel_file)
