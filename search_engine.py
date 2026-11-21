"""
MeaningMap - the NLP engine.

This file contains ALL the NLP logic of the project. The Streamlit app (app.py)
only calls these functions and draws the results.

Two ways of searching text are implemented and compared:

1. KEYWORD search  -> TF-IDF vectors + cosine similarity   (classic NLP, 1970s idea)
2. SEMANTIC search -> Sentence-BERT embeddings + cosine     (modern transformer NLP)
3. HYBRID search   -> a weighted mix of the two scores
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"

# Small, fast pretrained Sentence-BERT model (22 million parameters, 384-dim vectors).
# Hide a harmless Windows warning from the Hugging Face model cache.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# 1. Loading data and the model
# ---------------------------------------------------------------------------

def load_documents():
    """Read the knowledge base (80 short documents in 8 categories)."""
    return pd.read_csv(DATA_DIR / "documents.csv")


def load_eval_queries():
    """Read the hand-labelled test queries used for evaluation."""
    with open(DATA_DIR / "eval_queries.json", encoding="utf-8") as f:
        return json.load(f)


def load_model():
    """Download (first time only) and load the Sentence-BERT model."""
    from sentence_transformers import SentenceTransformer  # imported here because it is slow
    return SentenceTransformer(MODEL_NAME, device="cpu")


# ---------------------------------------------------------------------------
# 2. Building the two search indexes
# ---------------------------------------------------------------------------

def build_keyword_index(texts):
    """
    TF-IDF index.
    - lowercase=True       -> 'Sleep' and 'sleep' become the same word
    - stop_words='english' -> removes very common words like 'the', 'is', 'and'
    Each document becomes a long sparse vector with one number per vocabulary word.
    """
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix


def build_semantic_index(model, texts):
    """
    Sentence-BERT index.
    Each document becomes a dense 384-number vector that captures its MEANING.
    normalize_embeddings=True makes every vector length 1, so a simple dot
    product equals cosine similarity.
    """
    return model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)


# ---------------------------------------------------------------------------
# 3. Scoring a query against every document
# ---------------------------------------------------------------------------

def keyword_scores(query, vectorizer, tfidf_matrix):
    """Cosine similarity between the query's TF-IDF vector and each document."""
    query_vector = vectorizer.transform([query])
    return cosine_similarity(query_vector, tfidf_matrix)[0]


def semantic_scores(query, model, embeddings):
    """Cosine similarity between the query's embedding and each document."""
    query_vector = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    return embeddings @ query_vector


def min_max(scores):
    """Rescale scores to the range 0..1 so two different methods can be mixed."""
    low, high = scores.min(), scores.max()
    if high - low < 1e-9:  # all scores equal (e.g. no keyword matched at all)
        return np.zeros_like(scores)
    return (scores - low) / (high - low)


def hybrid_scores(kw_scores, sem_scores, alpha=0.7):
    """alpha = weight of meaning. alpha=1 -> pure semantic, alpha=0 -> pure keyword."""
    return alpha * min_max(sem_scores) + (1 - alpha) * min_max(kw_scores)


def top_k(scores, k=5):
    """Indexes of the k highest scores, best first."""
    return list(np.argsort(-scores)[:k])


# ---------------------------------------------------------------------------
# 4. Explanations ("why did this document match?")
# ---------------------------------------------------------------------------

def preprocessing_steps(text, vectorizer):
    """Show every TF-IDF preprocessing step for one piece of text."""
    lowered = vectorizer.build_preprocessor()(text)          # lower-casing
    tokens = vectorizer.build_tokenizer()(lowered)
    stop_words = vectorizer.get_stop_words()
    kept = [t for t in tokens if t not in stop_words]
    in_vocab = [t for t in kept if t in vectorizer.vocabulary_]

    vector = vectorizer.transform([text]).toarray()[0]
    vocab = vectorizer.get_feature_names_out()
    weights = {vocab[i]: round(float(vector[i]), 3) for i in np.nonzero(vector)[0]}

    return {
        "lowercased": lowered,
        "tokens": tokens,
        "removed_stopwords": [t for t in tokens if t in stop_words],
        "kept": kept,
        "in_vocabulary": in_vocab,
        "unknown_words": [t for t in kept if t not in vectorizer.vocabulary_],
        "tfidf_weights": dict(sorted(weights.items(), key=lambda x: -x[1])),
    }


def shared_keywords(query, doc_text, vectorizer):
    """Words (after preprocessing) that appear in BOTH the query and the document."""
    analyzer = vectorizer.build_analyzer()
    return sorted(set(analyzer(query)) & set(analyzer(doc_text)))


WORD_PATTERN = r"[A-Za-z0-9][A-Za-z0-9'-]*"


def word_meaning_scores(query, doc_text, model):
    """
    For every word in the document, how close is its meaning to the query?
    We embed each word on its own and compare it with the query embedding.
    Returns a dictionary {lower-case word: similarity}.
    """
    unique_words = sorted({w.lower() for w in re.findall(WORD_PATTERN, doc_text)})
    vectors = model.encode([query] + unique_words, normalize_embeddings=True, show_progress_bar=False)
    query_vec, word_vecs = vectors[0], vectors[1:]
    return dict(zip(unique_words, (word_vecs @ query_vec).tolist()))


def word_similarity_matrix(sentence_a, sentence_b, model):
    """Meaning similarity between every content word of A and every content word of B."""
    # analyzer = lowercase + tokenize + remove stop words
    analyzer = TfidfVectorizer(stop_words="english").build_analyzer()
    words_a = list(dict.fromkeys(analyzer(sentence_a)))
    words_b = list(dict.fromkeys(analyzer(sentence_b)))
    if not words_a or not words_b:
        return words_a, words_b, np.zeros((len(words_a), len(words_b)))
    vec_a = model.encode(words_a, normalize_embeddings=True, show_progress_bar=False)
    vec_b = model.encode(words_b, normalize_embeddings=True, show_progress_bar=False)
    return words_a, words_b, vec_a @ vec_b.T


def compare_two_sentences(sentence_a, sentence_b, model):
    """Similarity of two sentences, measured by TF-IDF and by Sentence-BERT."""
    # A fresh TF-IDF vectorizer fitted on just these two sentences, so that any
    # word the user types is part of the vocabulary.
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
    try:
        pair_matrix = vectorizer.fit_transform([sentence_a, sentence_b])
        keyword_sim = float(cosine_similarity(pair_matrix[0], pair_matrix[1])[0][0])
    except ValueError:  # both sentences contain only stop words -> nothing to compare
        keyword_sim = 0.0
    emb = model.encode([sentence_a, sentence_b], normalize_embeddings=True, show_progress_bar=False)
    semantic_sim = float(emb[0] @ emb[1])
    return keyword_sim, semantic_sim, shared_keywords(sentence_a, sentence_b, vectorizer)


# ---------------------------------------------------------------------------
# 5. 2-D maps of the vector spaces
# ---------------------------------------------------------------------------

def project_semantic_space(embeddings, query_embedding):
    """Squash 384-dim embeddings to 2-D with PCA so we can plot them."""
    pca = PCA(n_components=2, random_state=0).fit(embeddings)
    doc_xy = pca.transform(embeddings)
    query_xy = pca.transform(query_embedding.reshape(1, -1))[0]
    return doc_xy, query_xy, float(pca.explained_variance_ratio_.sum())


def project_keyword_space(tfidf_matrix, query_vector):
    """Same idea for sparse TF-IDF vectors (TruncatedSVD = PCA for sparse data)."""
    svd = TruncatedSVD(n_components=2, random_state=0).fit(tfidf_matrix)
    doc_xy = svd.transform(tfidf_matrix)
    query_xy = svd.transform(query_vector)[0]
    return doc_xy, query_xy, float(svd.explained_variance_ratio_.sum())


def category_separation(vectors_2d_or_full, categories):
    """
    Silhouette score: how well documents of the same category cluster together.
    Ranges from -1 to 1; higher means cleaner topic clusters.
    """
    from sklearn.metrics import silhouette_score
    return float(silhouette_score(vectors_2d_or_full, categories, metric="cosine"))


# ---------------------------------------------------------------------------
# 6. Evaluation with standard Information-Retrieval metrics
# ---------------------------------------------------------------------------

def rank_of_first_relevant(ranked_ids, relevant_ids):
    """1-based position of the first relevant document (None if not found)."""
    for position, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            return position
    return None


def evaluate(docs, eval_queries, model, vectorizer, tfidf_matrix, embeddings, alpha=0.7, k=5):
    """
    For each test query, rank all documents with each method and compute:
      - Precision@1 : is the top result relevant?          (1 or 0)
      - Recall@k    : fraction of relevant docs found in the top k
      - MRR         : 1 / rank of the first relevant document
    Returns (per_query_table, summary_table).
    """
    doc_ids = docs["id"].tolist()
    rows = []
    for item in eval_queries:
        query, relevant = item["query"], set(item["relevant"])
        kw = keyword_scores(query, vectorizer, tfidf_matrix)
        sem = semantic_scores(query, model, embeddings)
        methods = {"Keyword (TF-IDF)": kw, "Semantic (SBERT)": sem,
                   "Hybrid": hybrid_scores(kw, sem, alpha)}
        for method, scores in methods.items():
            order = np.argsort(-scores)
            if method == "Keyword (TF-IDF)":
                # A keyword score of 0 means "no shared word": the order among those documents
                # is arbitrary, so they are NOT counted as found.
                order = [i for i in order if scores[i] > 0]
            ranked_ids = [doc_ids[i] for i in order]
            rank = rank_of_first_relevant(ranked_ids, relevant)
            found_in_k = len(relevant & set(ranked_ids[:k]))
            rows.append({
                "query": query,
                "method": method,
                "rank_of_answer": rank,
                "precision@1": 1.0 if rank == 1 else 0.0,
                f"recall@{k}": found_in_k / len(relevant),
                "reciprocal_rank": 1.0 / rank if rank else 0.0,
            })
    per_query = pd.DataFrame(rows)
    summary = (per_query.groupby("method", sort=False)[["precision@1", f"recall@{k}", "reciprocal_rank"]]
               .mean().rename(columns={"reciprocal_rank": "MRR"}).round(3))
    return per_query, summary
