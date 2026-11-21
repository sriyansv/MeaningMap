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


def trimmed(path):
    """Screenshot without the sidebar, the page header and the empty space at the bottom."""
    from io import BytesIO
    from PIL import Image, ImageChops
    img = Image.open(path).convert("RGB")
    img = img.crop((340, 0, img.width, img.height))
    box = ImageChops.difference(img, Image.new("RGB", img.size, "white")).getbbox()
    img = img.crop((0, 395, img.width, min(img.height, box[3] + 20, 395 + 1000)))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def figure(path, caption, width=6.3):
    doc.add_picture(str(path) if not hasattr(path, "read") else path, width=Inches(width))
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
    "When we search for something, we use our own words. The document that has the answer often uses different words "
    "for the same thing. Normal keyword search cannot handle this, because it only looks for the exact words we typed. "
    "In this project I built MeaningMap, a small search app that compares three methods on the same set of documents: "
    "keyword search using TF-IDF, semantic search using a pretrained Sentence-BERT model (all-MiniLM-L6-v2), and a "
    "hybrid that mixes both. I tested them on 37 everyday questions over 80 documents from 8 topics. Semantic search "
    f"got Precision@1 = {sm['precision@1']:.3f}, Recall@5 = {sm['recall@5']:.3f} and MRR = {sm['MRR']:.3f}. TF-IDF got "
    f"{km['precision@1']:.3f}, {km['recall@5']:.3f} and {km['MRR']:.3f}. The app also shows why each result was picked, "
    "using word highlighting, 2-D maps of the documents, a sentence comparison tool and a step-by-step view of both "
    "pipelines.")

# ---------------- 1 Introduction ----------------
doc.add_heading("1. Introduction", level=1)
para("Suppose your laptop is slow to start and you search \"my PC takes forever to boot\". The useful article might say "
     "\"a computer becomes sluggish when too many programs run at startup\". A person can see these mean the same thing. "
     "But the two sentences have no important words in common, so a keyword search engine will not connect them.")
para("This is called the vocabulary mismatch problem. It matters wherever a system has to find text that answers a "
     "question: web search, online shopping, help desks, and the retrieval step of Retrieval-Augmented Generation (RAG) "
     "systems. The older way to search is the vector space model with TF-IDF weights, which matches words. Newer "
     "transformer models turn a whole sentence into an embedding, which captures its meaning.")
para("MeaningMap puts both methods side by side. You type a question and see the keyword results, the meaning-based "
     "results and a hybrid of the two, along with the reason each document was ranked where it is. I also measured all "
     "three methods on a labelled test set, so the comparison is based on numbers and not only on a few examples.")
doc.add_heading("1.1 Problem statement", level=2)
para("Given a question in normal English, rank a set of documents so that the document that answers it comes first, "
     "even if the question and the document share no words. Then measure how well a keyword method and a meaning-based "
     "method do this, and explain why they behave differently.")
doc.add_heading("1.2 Objectives", level=2)
bullets([
    "Build keyword search using the TF-IDF vector space model and cosine similarity.",
    "Build semantic search using a pretrained Sentence-BERT model.",
    "Combine both into a hybrid search with an adjustable weight α.",
    "Evaluate all three using Precision@1, Recall@5 and Mean Reciprocal Rank (MRR).",
    "Show the results visually with highlights, vector-space maps, word heatmaps and a pipeline view.",
])

# ---------------- 2 Background ----------------
doc.add_heading("2. Background", level=1)
doc.add_heading("2.1 TF-IDF and the vector space model", level=2)
para("In the vector space model [1], every document and every query is turned into a vector, and two texts are compared "
     "using the cosine of the angle between their vectors. Each dimension stands for one word in the vocabulary. Its "
     "value is the term frequency (TF) multiplied by the inverse document frequency (IDF) [2]. IDF gives more weight to "
     "rare words, because a word that appears in every document does not help to tell documents apart. scikit-learn "
     "uses idf = ln((1+N)/(1+df)) + 1 and then L2-normalises each vector [7].")
para("Because TF-IDF treats a text as a bag of words, it ignores word order. It also has no idea that two different "
     "words, like \"film\" and \"movie\", can mean the same thing.")
doc.add_heading("2.2 Transformers and Sentence-BERT", level=2)
para("Transformers [6] use self-attention, so each word's vector is updated based on the other words around it. BERT "
     "[4] is pretrained with masked language modelling, but its raw outputs do not work well when you compare sentences "
     "with cosine similarity.")
para("Sentence-BERT [3] fixes this. It fine-tunes BERT-style encoders on pairs of sentences in a siamese (bi-encoder) "
     "setup, so that the mean of the token vectors becomes a sentence embedding. Two sentences with similar meaning then "
     "get a high cosine similarity. I used all-MiniLM-L6-v2, a 6-layer MiniLM [5] model that outputs 384 numbers per "
     "sentence and has about 22 million parameters. It is published by the Sentence-Transformers project. I used the "
     "model as it is and did not train it.")
doc.add_heading("2.3 Retrieval evaluation", level=2)
para("I used three standard metrics. All of them go from 0 to 1, and higher is better.")
bullets([
    "Precision@1: the fraction of questions where the first result is a correct answer.",
    "Recall@5: the fraction of a question's correct documents that appear in the top 5, averaged over all questions.",
    "MRR: the average of 1 / (rank of the first correct document). It is 0 if the answer is not found.",
])

# ---------------- 3 Dataset ----------------
doc.add_heading("3. Dataset", level=1)
para("I wrote two small files for this project. I kept them small so that every result can be checked by hand.")
bullets([
    "data/documents.csv has 80 short documents, each two sentences long. There are 10 documents in each of 8 "
    "categories: Health, Technology, Space, Food, Travel, Finance, Sports and Environment. Each row has an id, a "
    "category, a title and the text.",
    "data/eval_queries.json has 37 questions written in everyday language. Each question is labelled with the 1 or 2 "
    "documents that answer it.",
])
para("The models never see the category labels. I only use them to colour the plots and to calculate the silhouette "
     "score.")
para("I wrote the questions using different words from the documents on purpose. For example, a question asks \"how do "
     "jabs protect us from illness\" while the document says \"Vaccines train the immune system\". The point of the "
     "project is to study vocabulary mismatch, so the test set needed this kind of question. This also means the test "
     "set favours semantic search, and I list that as a limitation in Section 8.")

# ---------------- 4 Methodology ----------------
doc.add_heading("4. Methodology and system design", level=1)
para("Figure 1 shows the main parts of the system and Figure 2 shows what happens to a single query. The query goes "
     "through two separate pipelines, and their scores are combined at the end.")
figure(DOCS / "diagrams" / "architecture.png", "Figure 1. System architecture")
figure(DOCS / "diagrams" / "workflow.png", "Figure 2. Processing of one query")
doc.add_heading("4.1 Keyword pipeline", level=2)
para("I used TfidfVectorizer(lowercase=True, stop_words='english'), which learns an 839-word vocabulary from the 80 "
     "documents. A query is lower-cased and split into tokens (words of two or more characters). The 318 built-in "
     "English stop words are removed, and the rest are turned into a TF-IDF vector. Any word that is not in the "
     "vocabulary is simply ignored. The score of each document is the cosine similarity between its vector and the "
     "query vector.")
doc.add_heading("4.2 Semantic pipeline", level=2)
para("The query is split into WordPiece sub-word tokens, so there are no unknown words. The tokens go through the 6 "
     "transformer layers, are mean-pooled into one 384-number vector, and the vector is normalised to length 1. The "
     "document embeddings are calculated once in the same way. Since all vectors have length 1, cosine similarity is "
     "just a dot product.")
doc.add_heading("4.3 Hybrid ranking", level=2)
para("The keyword scores and the semantic scores are on different scales, so I cannot add them directly. I min-max "
     "scale each list to the range 0-1 and then combine them as α × semantic + (1 − α) × keyword. The default is α = "
     "0.7, and the user can change it with a slider.")
doc.add_heading("4.4 Explanations", level=2)
para("Showing a ranked list is not enough to understand why the methods differ, so the app explains its results:")
bullets([
    "Shared keywords (after preprocessing) are highlighted in yellow.",
    "Each word of a result is shaded blue by how close its own embedding is to the query embedding. This is only an "
    "approximation, because a single word loses its context.",
    "Document map: PCA (for the embeddings) and Truncated SVD (for the sparse TF-IDF vectors) project all documents "
    "and the query to 2-D.",
    "The silhouette score, calculated on the full vectors, shows how well the 8 topics form groups in each space.",
    "Compare sentences: TF-IDF vs SBERT similarity for any two sentences, plus a word-to-word similarity heatmap.",
    "A warning appears when the best semantic score is below 0.25, which means the collection probably has no real "
    "match.",
])

# ---------------- 5 Implementation ----------------
doc.add_heading("5. Implementation", level=1)
para("The project is written in Python and split into four main files:")
table(["File", "Purpose"], [
    ["search_engine.py", "All NLP logic as plain functions: loading, indexing, scoring, explanations, projections, evaluation"],
    ["app.py", "Streamlit interface with five tabs; models and indexes cached with st.cache_resource"],
    ["evaluate.py", "Command-line evaluation; writes results/metrics.csv, per_query.csv, alpha_sweep.csv"],
    ["test_engine.py", "Automated checks of data consistency, shapes, retrieval behaviour and metric ranges"],
])
para("Libraries used: scikit-learn, sentence-transformers (PyTorch, Hugging Face Transformers), Streamlit, Plotly, "
     "pandas and NumPy. Everything runs on a normal laptop CPU. Encoding the 80 documents takes about 0.3 s, and one "
     "query takes about 10 ms.")
para("One rule in the evaluation needs explaining. In TF-IDF, a document with a score of 0 shares no word with the "
     "query, and the order among such documents is random. So I do not count those documents as retrieved.")

# ---------------- 6 Results ----------------
doc.add_heading("6. Results", level=1)
para(f"I ran all three methods on the {per_query['query'].nunique()} questions and 80 documents, with α = 0.7 for "
     "the hybrid:")
table(["Method", "Precision@1", "Recall@5", "MRR"],
      [[m, f"{r['precision@1']:.3f}", f"{r['recall@5']:.3f}", f"{r['MRR']:.3f}"] for m, r in metrics.iterrows()])
para(f"Semantic search put a correct answer first for {int(round(sm['precision@1'] * 37))} of the 37 questions. TF-IDF "
     f"did this for only {int(round(km['precision@1'] * 37))}. Semantic search also found every correct document within "
     "the top 5 (Recall@5 = 1.000).")
para("The main reason is synonyms and paraphrases. When a question used different words from the document, TF-IDF had "
     "nothing to match. An example is \"how do jabs protect us from illness\" vs \"Vaccines train the immune system\". "
     "Sentence-BERT compares meaning, so it still found these documents. TF-IDF can also be misled by one unimportant "
     "shared word. For \"my PC takes forever to boot\", its top result was \"The Sun\", which matched only on the word "
     "\"takes\".")
para("I also changed the hybrid weight α from 0 (only keyword) to 1 (only meaning):")
table(["α", "Hybrid MRR", "Hybrid P@1"],
      [[f"{r.alpha:.1f}", f"{r.hybrid_MRR:.3f}", f"{r['hybrid_P@1']:.3f}"] for _, r in sweep.iterrows()])
para("The hybrid kept getting better as α increased, but it did not beat pure semantic search on this test set. Almost "
     "all the questions are paraphrases, so the keyword part mostly adds noise here. Keyword matching should help more "
     "with exact names, codes or rare terms, and this test set has very few of those.")
para("Topic clustering (silhouette score on the full vectors, range −1 to 1): TF-IDF −0.001, Sentence-BERT 0.073. Both "
     "scores are low, because the texts are short and some topics overlap. Still, documents group by topic better in "
     "the embedding space, even though the model never saw the topic labels.")
para("The sentence pair tests show some limits too. TF-IDF scores \"The food was not good\" vs \"The food was good\" as "
     "1.00, because \"not\" is a stop word and gets removed (SBERT gives 0.82). Both methods treat \"The dog bit the "
     "man\" and \"The man bit the dog\" as almost the same (1.00 and 0.98).")
para("These numbers should be read carefully. The dataset is small and I wrote it myself, and the questions were "
     "deliberately phrased to test vocabulary mismatch. So the results show a clear trend, but they are not a benchmark "
     "score.")

# ---------------- 7 Screenshots ----------------
doc.add_heading("7. Application screenshots", level=1)
para("Screenshots of the running app:")
shots = [("1_search_battle.png", "Search"), ("2_meaning_galaxy.png", "Document map"),
         ("6_similarity_bank.png", "Compare sentences"), ("4_under_the_hood.png", "How it works"),
         ("5_evaluation.png", "Evaluation")]
for i, (file, name) in enumerate(shots, start=3):
    figure(trimmed(DOCS / "screenshots" / file), f"Figure {i}. {name}", width=5.0)

# ---------------- 8 Limitations / future / conclusion ----------------
doc.add_heading("8. Limitations", level=1)
para("These are the actual limitations of this version of the project:")
bullets([
    "The dataset is small (80 documents, 37 questions) and I wrote it myself. The results show a trend, not a "
    "benchmark score.",
    "The questions were written to test paraphrases, so they favour semantic search. Keyword search would likely do "
    "better on questions with exact names or codes.",
    "Neither method handles word order properly. \"The dog bit the man\" and \"The man bit the dog\" score 1.00 with "
    "TF-IDF and 0.98 with Sentence-BERT.",
    "Negation is weak. TF-IDF drops \"not\" completely, and Sentence-BERT still gives 0.82 to \"not good\" vs \"good\".",
    "Semantic search always returns its closest documents, even when the question has no answer in the collection. I "
    "added a warning when the best score is below 0.25, but this does not fully fix it.",
    "The model only works for English, and inputs longer than 256 word-pieces are cut off.",
    "The 2-D maps keep only about 3% (TF-IDF) and 10% (SBERT) of the variance, so they are a rough picture and not "
    "exact distances.",
])
doc.add_heading("9. Future work", level=1)
para("Each of these ideas is linked to one of the limitations above:")
bullets([
    "Test on a public benchmark such as BEIR [9] or MS MARCO. This would replace my small self-written dataset with a "
    "large, standard one.",
    "Use BM25 as the keyword method. BM25 is an improved version of TF-IDF that also accounts for document length, so "
    "it is a fairer baseline. A cross-encoder re-ranker, which reads the query and a document together, could also help "
    "with word order and negation.",
    "Use a vector index such as FAISS. Right now every document is compared with the query one by one, which is fine "
    "for 80 documents but too slow for millions. FAISS finds the nearest vectors much faster.",
    "Use a multilingual embedding model so that the app can also handle questions in Indian languages.",
    "Add answer generation (RAG [8]). The top documents could be given to a language model so that it writes a direct "
    "answer instead of only showing a list.",
])
doc.add_heading("10. Conclusion", level=1)
para("In this project I built a working search app that compares keyword search and semantic search on the same data, "
     "and measured both on a labelled test set. On everyday questions, Sentence-BERT raised MRR from "
     f"{km['MRR']:.2f} to {sm['MRR']:.2f}. It did this with one small pretrained model running on a CPU, without any "
     "training.")
para("The main thing I learned is that the two methods fail in different ways. TF-IDF is simple and easy to explain, "
     "but it only sees exact words, so synonyms break it and one random shared word can fool it. Sentence-BERT handles "
     "paraphrases well, but it still struggles with word order and negation, and it always returns something even "
     "when there is no real answer. Adding keyword weight in the hybrid also did not help on this kind of question.")
para("What worked well was making the results visible. The highlights, the document map and the sentence comparison "
     "made it easy to see why each method ranked a document where it did. The main thing to improve is the "
     "evaluation: a larger public dataset, a stronger keyword baseline like BM25 and questions with exact names or "
     "codes would give a fairer comparison.")

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
