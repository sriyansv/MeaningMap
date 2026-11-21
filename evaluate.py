"""
Run the evaluation from the command line and save the results.

    python evaluate.py

Prints Precision@1, Recall@5 and MRR for Keyword, Semantic and Hybrid search,
and writes results/metrics.csv and results/per_query.csv.
"""

import pandas as pd

import search_engine as se

docs = se.load_documents()
queries = se.load_eval_queries()
model = se.load_model()

vectorizer, tfidf_matrix = se.build_keyword_index(docs["text"])
embeddings = se.build_semantic_index(model, docs["text"])

per_query, summary = se.evaluate(docs, queries, model, vectorizer, tfidf_matrix, embeddings)

print(f"\nDocuments: {len(docs)}   Test queries: {len(queries)}\n")
print(summary.to_string())

# How cleanly do the 8 topics cluster in each vector space? (silhouette, -1..1)
print("\nTopic clustering (silhouette score, higher = cleaner clusters)")
print("  TF-IDF space :", round(se.category_separation(tfidf_matrix.toarray(), docs["category"]), 3))
print("  SBERT space  :", round(se.category_separation(embeddings, docs["category"]), 3))

# How does the hybrid weight alpha change the result? (alpha = weight of meaning)
print("\nHybrid alpha sweep (MRR)")
sweep = []
for alpha in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    _, s = se.evaluate(docs, queries, model, vectorizer, tfidf_matrix, embeddings, alpha=alpha)
    sweep.append({"alpha": alpha, "hybrid_MRR": s.loc["Hybrid", "MRR"], "hybrid_P@1": s.loc["Hybrid", "precision@1"]})
    print(f"  alpha={alpha:.1f}  MRR={s.loc['Hybrid', 'MRR']:.3f}  P@1={s.loc['Hybrid', 'precision@1']:.3f}")

results_dir = se.PROJECT_DIR / "results"
results_dir.mkdir(exist_ok=True)
summary.to_csv(results_dir / "metrics.csv")
per_query.to_csv(results_dir / "per_query.csv", index=False)
pd.DataFrame(sweep).to_csv(results_dir / "alpha_sweep.csv", index=False)
print(f"\nSaved results to {results_dir}")
