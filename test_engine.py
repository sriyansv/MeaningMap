"""
Quick automated checks of the NLP engine.
    python test_engine.py
"""
import numpy as np

import search_engine as se

docs = se.load_documents()
queries = se.load_eval_queries()
model = se.load_model()
vectorizer, tfidf_matrix = se.build_keyword_index(docs["text"])
embeddings = se.build_semantic_index(model, docs["text"])

# 1. Data is consistent
assert len(docs) == 80 and docs["id"].is_unique, "documents.csv should have 80 unique ids"
all_ids = set(docs["id"])
for q in queries:
    assert set(q["relevant"]) <= all_ids, f"unknown id in eval query: {q}"

# 2. Shapes and normalisation
assert embeddings.shape == (80, 384)
assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-4)

# 3. Keyword search finds an exact-word query
kw = se.keyword_scores("solar panels electricity", vectorizer, tfidf_matrix)
assert docs.iloc[se.top_k(kw, 1)[0]]["id"] == "EN04"

# 4. Semantic search finds a paraphrase with NO shared words
query = "what killed the dinosaurs"
sem = se.semantic_scores(query, model, embeddings)
assert docs.iloc[se.top_k(sem, 1)[0]]["id"] == "SP07"

# 5. A query made only of unknown words gives all-zero keyword scores (no crash)
kw_zero = se.keyword_scores("xyzzy blorft", vectorizer, tfidf_matrix)
assert kw_zero.max() == 0
assert se.hybrid_scores(kw_zero, sem).max() <= 1.0

# 6. Similarity lab behaves as expected
kw_sim, sem_sim, shared = se.compare_two_sentences("The movie was fantastic", "I really enjoyed the film", model)
assert kw_sim == 0 and sem_sim > 0.4, (kw_sim, sem_sim)
kw_sim, sem_sim, _ = se.compare_two_sentences("the", "is", model)   # only stop words
assert kw_sim == 0

# 7. Explanations
steps = se.preprocessing_steps("Why does my PC take forever to boot?", vectorizer)
assert "why" in steps["removed_stopwords"]
pairs = se.word_meaning_scores(query, docs.iloc[0]["text"], model)
assert len(pairs) > 0

# 8. Evaluation runs and metrics are within 0..1
per_query, summary = se.evaluate(docs, queries, model, vectorizer, tfidf_matrix, embeddings)
assert len(per_query) == 3 * len(queries)
assert ((summary >= 0) & (summary <= 1)).all().all()

print("All tests passed.\n")
print(summary.to_string())
