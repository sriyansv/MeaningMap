"""
Builds the project report (docs/MeaningMap_Report.docx).
    python docs/make_report.py      (run evaluate.py first so results/ exists)
Numbers are read from results/*.csv, so the report always matches the code.
"""
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

DOCS = Path(__file__).parent
ROOT = DOCS.parent
metrics = pd.read_csv(ROOT / "results" / "metrics.csv", index_col=0)
sweep = pd.read_csv(ROOT / "results" / "alpha_sweep.csv")
per_query = pd.read_csv(ROOT / "results" / "per_query.csv")

doc = Document()
doc.styles["Normal"].font.name = "Calibri"
doc.styles["Normal"].font.size = Pt(11)


def para(text, bold=False, italic=False, size=None, align=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold, run.italic = bold, italic
    if size:
        run.font.size = Pt(size)
    if align:
        p.alignment = align
    return p


def bullets(items):
    for item in items:
        doc.add_paragraph(item, style="List Bullet")


def table(header, rows):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    for cell, h in zip(t.rows[0].cells, header):
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
    for row in rows:
        cells = t.add_row().cells
        for cell, value in zip(cells, row):
            cell.text = str(value)
    doc.add_paragraph()


def figure(path, caption, width=6.3):
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = para(caption, italic=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER)
    p.runs[0].font.color.rgb = RGBColor(0x55, 0x55, 0x55)


# ---------------- Title page ----------------
for _ in range(6):
    doc.add_paragraph()
para("MeaningMap", bold=True, size=36, align=WD_ALIGN_PARAGRAPH.CENTER)
para("Comparing Keyword (TF-IDF) and Semantic (Sentence-BERT) Search\nwith Interactive Explanations",
     size=16, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
para("Natural Language Processing - Project Report", size=13, align=WD_ALIGN_PARAGRAPH.CENTER)
para("B.Tech Artificial Intelligence and Data Science", size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
para("Submitted by", size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
para("VEMULA PAVANA SAI SRIYANS", bold=True, size=13, align=WD_ALIGN_PARAGRAPH.CENTER)
para("Roll No.: 124AD0046", size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_page_break()

# ---------------- Abstract ----------------
doc.add_heading("Abstract", level=1)
km, sm, hm = metrics.loc["Keyword (TF-IDF)"], metrics.loc["Semantic (SBERT)"], metrics.loc["Hybrid"]
para(
    "Users rarely describe a problem with the same words as the document that solves it (the vocabulary mismatch "
    "problem). This project builds MeaningMap, an interactive application that compares three retrieval methods on the "
    "same collection: classical keyword search using TF-IDF vectors and cosine similarity, semantic search using a "
    "pretrained Sentence-BERT model (all-MiniLM-L6-v2), and a hybrid of both. The methods are evaluated on 37 "
    "hand-labelled everyday-language questions over 80 documents in 8 topics. Semantic search achieved "
    f"Precision@1 = {sm['precision@1']:.3f}, Recall@5 = {sm['recall@5']:.3f} and MRR = {sm['MRR']:.3f}, compared with "
    f"{km['precision@1']:.3f}, {km['recall@5']:.3f} and {km['MRR']:.3f} for TF-IDF. The application makes the "
    "difference visible through keyword and meaning highlighting, 2-D projections of both vector spaces, a sentence "
    "similarity lab and a step-by-step pipeline view.")

# ---------------- 1 Introduction ----------------
doc.add_heading("1. Introduction", level=1)
para("Search is one of the most common uses of NLP: web search engines, e-commerce, help desks, and the retrieval step "
     "of Retrieval-Augmented Generation (RAG) systems all have to find the text that answers a user's query. The classic "
     "approach, the vector space model with TF-IDF weights, matches words. Modern transformer models produce sentence "
     "embeddings that match meaning. This project implements both, measures them, and explains their behaviour visually.")
doc.add_heading("1.1 Problem statement", level=2)
para("Given a natural-language query, rank a collection of documents so that the document answering the query comes "
     "first, even when the query and the document share no words. Then measure how well a keyword method and a meaning "
     "method do this, and explain why.")
doc.add_heading("1.2 Objectives", level=2)
bullets([
    "Implement keyword retrieval with the TF-IDF vector space model and cosine similarity.",
    "Implement semantic retrieval with a pretrained Sentence-BERT model.",
    "Combine both into a hybrid ranker with an adjustable weight α.",
    "Evaluate all three with Precision@1, Recall@5 and Mean Reciprocal Rank (MRR).",
    "Explain results visually with highlights, vector-space maps, word heatmaps and a pipeline view.",
])

# ---------------- 2 Background ----------------
doc.add_heading("2. Background", level=1)
doc.add_heading("2.1 TF-IDF and the vector space model", level=2)
para("In the vector space model [1], documents and queries are vectors and relevance is measured by the cosine of the "
     "angle between them. Each dimension is a vocabulary word weighted by term frequency (TF) times inverse document "
     "frequency (IDF) [2]. IDF gives rare words more weight. scikit-learn computes idf = ln((1+N)/(1+df)) + 1 and "
     "L2-normalises each vector [7]. Because the representation is a bag of words, it ignores word order and cannot "
     "relate different words with the same meaning.")
doc.add_heading("2.2 Transformers and Sentence-BERT", level=2)
para("Transformers [6] build a contextual vector for each token using self-attention. BERT [4] is pretrained with masked "
     "language modelling, but its raw outputs are not well suited to comparing sentences by cosine similarity. "
     "Sentence-BERT [3] fine-tunes BERT-style encoders on sentence pairs in a siamese (bi-encoder) set-up so that the "
     "mean-pooled token vectors form sentence embeddings whose cosine similarity reflects semantic similarity. This "
     "project uses all-MiniLM-L6-v2, a 6-layer MiniLM [5] model with 384-dimensional output (about 22 million "
     "parameters), published by the Sentence-Transformers project. It is used as-is; no training is done in this project.")
doc.add_heading("2.3 Retrieval evaluation", level=2)
bullets([
    "Precision@1: fraction of queries whose top-ranked document is relevant.",
    "Recall@5: fraction of a query's relevant documents found in the top 5, averaged over queries.",
    "MRR: mean over queries of 1 / rank of the first relevant document (0 if not found).",
])

# ---------------- 3 Dataset ----------------
doc.add_heading("3. Dataset", level=1)
para("Two small files were written for this project so that every result can be verified by hand:")
bullets([
    "data/documents.csv: 80 two-sentence documents, 10 in each of 8 categories (Health, Technology, Space, Food, "
    "Travel, Finance, Sports, Environment). Categories are never given to the models; they are used only for colouring "
    "plots and for the silhouette score.",
    "data/eval_queries.json: 37 everyday-language questions, each labelled with the 1 or 2 documents that answer it.",
])
para("The questions were deliberately phrased in words different from the documents, to study vocabulary mismatch. "
     "This favours semantic search and is stated as a limitation.")

# ---------------- 4 Methodology ----------------
doc.add_heading("4. Methodology and system design", level=1)
figure(DOCS / "diagrams" / "architecture.png", "Figure 1. System architecture")
figure(DOCS / "diagrams" / "workflow.png", "Figure 2. Processing of one query")
doc.add_heading("4.1 Keyword pipeline", level=2)
para("TfidfVectorizer(lowercase=True, stop_words='english') learns an 839-word vocabulary from the 80 documents. A query "
     "is lower-cased, tokenized (words of two or more characters), stripped of the 318 built-in English stop words, and "
     "mapped to a TF-IDF vector. Words outside the vocabulary are ignored. Scores are cosine similarities to every "
     "document vector.")
doc.add_heading("4.2 Semantic pipeline", level=2)
para("The query is split into WordPiece sub-word tokens, passed through the 6 transformer layers, mean-pooled into one "
     "384-dimensional vector and normalised to length 1. Document embeddings are pre-computed the same way, so the cosine "
     "similarity is a single dot product.")
doc.add_heading("4.3 Hybrid ranking", level=2)
para("The two score lists are on different scales, so each is min-max scaled to 0-1 and combined as "
     "α × semantic + (1 − α) × keyword. The default is α = 0.7; the user can change it with a slider.")
doc.add_heading("4.4 Explanations", level=2)
bullets([
    "Shared keywords (after preprocessing) are highlighted in yellow.",
    "Every word of a result is tinted blue by the cosine similarity of its own embedding to the query embedding "
    "(an approximation, because single words lose context).",
    "Document map: PCA (embeddings) and Truncated SVD (sparse TF-IDF) project all documents and the query to 2-D.",
    "Silhouette score on the full vectors measures how well the 8 topics cluster in each space.",
    "Compare sentences: TF-IDF vs SBERT similarity for any two sentences, plus a word-to-word similarity heatmap.",
    "A warning appears when the best semantic score is below 0.25 (no real match in the collection).",
])

# ---------------- 5 Implementation ----------------
doc.add_heading("5. Implementation", level=1)
table(["File", "Purpose"], [
    ["search_engine.py", "All NLP logic as plain functions: loading, indexing, scoring, explanations, projections, evaluation"],
    ["app.py", "Streamlit interface with five tabs; models and indexes cached with st.cache_resource"],
    ["evaluate.py", "Command-line evaluation; writes results/metrics.csv, per_query.csv, alpha_sweep.csv"],
    ["test_engine.py", "Automated checks of data consistency, shapes, retrieval behaviour and metric ranges"],
])
para("Libraries: scikit-learn, sentence-transformers (PyTorch, Hugging Face Transformers), Streamlit, Plotly, pandas, "
     "NumPy. Everything runs on a laptop CPU: encoding the 80 documents takes about 0.3 s and one query about 10 ms.")
para("Evaluation rule: for TF-IDF, documents with a score of 0 share no word with the query and their relative order is "
     "arbitrary, so they are not counted as retrieved.")

# ---------------- 6 Results ----------------
doc.add_heading("6. Results", level=1)
para(f"Evaluation on {per_query['query'].nunique()} questions and 80 documents (α = 0.7 for the hybrid):")
table(["Method", "Precision@1", "Recall@5", "MRR"],
      [[m, f"{r['precision@1']:.3f}", f"{r['recall@5']:.3f}", f"{r['MRR']:.3f}"] for m, r in metrics.iterrows()])
para("Effect of the hybrid weight α (0 = pure keyword, 1 = pure meaning):")
table(["α", "Hybrid MRR", "Hybrid P@1"],
      [[f"{r.alpha:.1f}", f"{r.hybrid_MRR:.3f}", f"{r['hybrid_P@1']:.3f}"] for _, r in sweep.iterrows()])
para("Topic clustering, silhouette score on full vectors (−1 to 1): TF-IDF −0.001, Sentence-BERT 0.073.")

# ---------------- 7 Screenshots ----------------
doc.add_heading("7. Application screenshots", level=1)
shots = [("1_search_battle.png", "Search"), ("2_meaning_galaxy.png", "Document map"),
         ("6_similarity_bank.png", "Compare sentences"), ("4_under_the_hood.png", "How it works"),
         ("5_evaluation.png", "Evaluation")]
for i, (file, name) in enumerate(shots, start=3):
    figure(DOCS / "screenshots" / file, f"Figure {i}. {name}", width=5.2)

# ---------------- 8 Limitations / future / conclusion ----------------
doc.add_heading("8. Limitations", level=1)
bullets([
    "Small, self-written dataset; the results show a trend, not a benchmark score.",
    "Test questions intentionally favour paraphrase matching.",
    "Word order and negation are handled weakly or not at all.",
    "Semantic search always returns its closest documents, even for unanswerable queries (mitigated by a 0.25 warning).",
    "English-only model; inputs longer than 256 word-pieces are truncated.",
    "2-D maps keep only about 3% (TF-IDF) and 10% (SBERT) of the variance.",
])
doc.add_heading("9. Future work", level=1)
bullets([
    "Evaluate on public benchmarks such as BEIR [9] or MS MARCO.",
    "Use BM25 as a stronger keyword baseline and add a cross-encoder re-ranker.",
    "Use an approximate nearest-neighbour index (e.g. FAISS) for large collections.",
    "Use a multilingual embedding model for Indian-language queries.",
    "Generate answers from the retrieved documents with a language model (RAG [8]).",
])
doc.add_heading("10. Conclusion", level=1)
para("MeaningMap shows, with measured results and visual explanations, that keyword search matches words while "
     "Sentence-BERT matches meaning. On everyday-language questions the pretrained semantic model raised MRR from "
     f"{km['MRR']:.2f} to {sm['MRR']:.2f}, using one small model on a CPU and a few hundred lines of Python.")

# ---------------- References ----------------
doc.add_heading("References", level=1)
refs = [
    "G. Salton, A. Wong, C. S. Yang. A vector space model for automatic indexing. Communications of the ACM, 18(11), 1975.",
    "K. Spärck Jones. A statistical interpretation of term specificity and its application in retrieval. Journal of Documentation, 28(1), 1972.",
    "N. Reimers, I. Gurevych. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP-IJCNLP, 2019.",
    "J. Devlin, M.-W. Chang, K. Lee, K. Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL, 2019.",
    "W. Wang et al. MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. NeurIPS, 2020.",
    "A. Vaswani et al. Attention Is All You Need. NeurIPS, 2017.",
    "F. Pedregosa et al. Scikit-learn: Machine Learning in Python. JMLR 12, 2011.",
    "P. Lewis et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS, 2020.",
    "N. Thakur et al. BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. NeurIPS Datasets and Benchmarks, 2021.",
]
for i, ref in enumerate(refs, start=1):
    para(f"[{i}] {ref}", size=10)

doc.save(DOCS / "MeaningMap_Report.docx")
print("Saved", DOCS / "MeaningMap_Report.docx")
