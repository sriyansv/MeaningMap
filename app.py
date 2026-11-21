"""
MeaningMap - Streamlit user interface.

Run with:   streamlit run app.py

All NLP work happens in search_engine.py. This file only:
  1. loads the data / model once (cached),
  2. reads what the user types,
  3. calls the engine functions,
  4. draws cards, highlights and charts.
"""

import html
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import search_engine as se

st.set_page_config(page_title="MeaningMap", layout="wide")

# Colours (one fixed colour per method and per category, used everywhere)
METHOD_COLORS = {"Keyword (TF-IDF)": "#d95926", "Semantic (SBERT)": "#3987e5", "Hybrid": "#199e70"}
CATEGORY_COLORS = {
    "Health": "#3987e5", "Technology": "#d95926", "Space": "#199e70", "Food": "#c98500",
    "Travel": "#d55181", "Finance": "#0ca30c", "Sports": "#9085e9", "Environment": "#e66767",
    "Custom": "#3987e5",
}
CATEGORY_SYMBOLS = {
    "Health": "circle", "Technology": "square", "Space": "diamond", "Food": "triangle-up",
    "Travel": "pentagon", "Finance": "hexagon", "Sports": "triangle-down", "Environment": "cross",
    "Custom": "circle",
}

# Page styling (pure CSS - no logic here)
st.markdown("""
<style>
.block-container { padding-top: 2rem; max-width: 1300px; }
.col-head { font-size: 1.05rem; font-weight: 600; padding: 6px 0 6px 10px; margin-bottom: 10px;
            border-left: 4px solid var(--c); }
.col-head small { display: block; font-weight: 400; color: #666; font-size: .8rem; }
.card { border: 1px solid #ddd; border-radius: 6px; padding: 12px 14px; margin-bottom: 10px; background: #fff; }
.card.correct { border: 2px solid #199e70; }
.card-top { display: flex; align-items: center; gap: 8px; font-size: .8rem; color: #666; }
.rank { font-weight: 600; color: #222; }
.badge { padding: 1px 6px; border-radius: 3px; font-size: .72rem; color: #333; border: 1px solid var(--c); }
.score { margin-left: auto; font-family: monospace; color: #222; }
.title { font-weight: 600; margin: 4px 0; color: #111; }
.bar { height: 4px; background: #eee; margin-bottom: 8px; }
.bar div { height: 100%; }
.text { font-size: .88rem; line-height: 1.55; color: #333; }
.why { font-size: .75rem; color: #777; margin-top: 6px; }
.kw { background: #ffe58a; padding: 0 2px; }
.verdict { padding: 10px 14px; margin: 4px 0 14px 0; background: #f4f6f8; border-left: 4px solid #3987e5; font-size: .95rem; }
.chip { display: inline-block; padding: 2px 7px; margin: 2px; border-radius: 3px; font-size: .82rem;
        background: #f1f1f1; border: 1px solid #ccc; font-family: monospace; }
.chip.gone { opacity: .4; text-decoration: line-through; }
.chip.sub { border-color: #3987e5; }
.step { font-weight: 600; margin-top: 12px; }
.big { font-size: 2.2rem; font-weight: 600; line-height: 1.1; }
.big-label { color: #666; font-size: .85rem; }
.metric-card { border: 1px solid #ddd; border-radius: 6px; padding: 14px; text-align: center; border-top: 3px solid var(--c); }
</style>
""", unsafe_allow_html=True)


# Load model + build indexes ONCE (Streamlit caches the results)
@st.cache_resource(show_spinner="Loading the Sentence-BERT model (first run downloads ~90 MB)...")
def get_model():
    return se.load_model()


@st.cache_resource(show_spinner="Building the TF-IDF and embedding indexes...")
def get_indexes(texts):
    vectorizer, tfidf_matrix = se.build_keyword_index(list(texts))
    embeddings = se.build_semantic_index(get_model(), list(texts))
    return vectorizer, tfidf_matrix, embeddings


@st.cache_data(show_spinner="Evaluating all three methods on the test queries...")
def get_evaluation(alpha):
    docs = se.load_documents()
    vectorizer, tfidf_matrix, embeddings = get_indexes(tuple(docs["text"]))
    return se.evaluate(docs, se.load_eval_queries(), get_model(), vectorizer, tfidf_matrix, embeddings, alpha=alpha)


try:
    model = get_model()
except Exception as error:  # e.g. no internet on the very first run
    st.error(f"Could not load the language model `{se.MODEL_NAME}`.\n\n"
             f"The first run needs internet to download it once. Details: {error}")
    st.stop()

# Sidebar - settings
with st.sidebar:
    st.markdown("### Settings")
    corpus_choice = st.radio("Document collection", ["Built-in knowledge base (80 docs)", "Paste my own documents"])
    if corpus_choice.startswith("Paste"):
        pasted = st.text_area("One document per line (at least 3)", height=220, value=(
            "The stock market crashed after the bank announced huge losses.\n"
            "Our football team won the championship in the final minute.\n"
            "Doctors recommend eating more vegetables and sleeping well.\n"
            "The new phone has a faster processor and a better camera.\n"
            "Heavy rain flooded several villages near the river."))
        lines = [line.strip() for line in pasted.splitlines() if line.strip()]
        if len(lines) < 3:
            st.warning("Please enter at least 3 documents. Showing the built-in collection meanwhile.")
            docs = se.load_documents()
        else:
            docs = pd.DataFrame({"id": [f"U{i+1:02d}" for i in range(len(lines))], "category": "Custom",
                                 "title": [f"Document {i+1}" for i in range(len(lines))], "text": lines})
    else:
        docs = se.load_documents()

    alpha = st.slider("Hybrid: weight of meaning (α)", 0.0, 1.0, 0.7, 0.05,
                      help="Hybrid score = α × semantic + (1 − α) × keyword (both rescaled to 0–1)")
    top_n = st.slider("Results per method", 3, 8, 4)
    show_heat = st.toggle("Colour words by meaning-similarity", value=True)
    st.markdown("---")
    st.caption(f"**Embedding model:** `{se.MODEL_NAME}`  \n384-dim vectors · 22M parameters · runs on CPU")
    st.caption(f"**Collection:** {len(docs)} documents")

vectorizer, tfidf_matrix, embeddings = get_indexes(tuple(docs["text"]))
eval_queries = se.load_eval_queries()
expected_answers = {item["query"].lower(): set(item["relevant"]) for item in eval_queries}

# Header
st.title("MeaningMap")
st.write("Comparing keyword search (TF-IDF) with semantic search (Sentence-BERT) on the same set of documents. "
         "Type a question and see which method finds the right answer.")


# Small helpers for drawing
def decorate_words(text, decorate):
    """Rebuild the text, letting decorate(word) wrap each word in HTML. Spacing and punctuation stay unchanged."""
    result, last_end = [], 0
    for match in re.finditer(se.WORD_PATTERN, text):
        result.append(html.escape(text[last_end:match.start()]))  # spaces / punctuation before the word
        result.append(decorate(match.group()))
        last_end = match.end()
    result.append(html.escape(text[last_end:]))
    return "".join(result)


def highlight_keywords(text, keywords):
    """Yellow background on every word that the query and document share."""
    analyzer = vectorizer.build_analyzer()

    def decorate(word):
        is_match = any(token in keywords for token in analyzer(word))
        return f'<span class="kw">{html.escape(word)}</span>' if is_match else html.escape(word)
    return decorate_words(text, decorate)


def highlight_meaning(text, word_scores):
    """Blue background whose strength shows how close each word's meaning is to the query."""
    def decorate(word):
        sim = word_scores.get(word.lower(), 0.0)
        strength = float(np.clip((sim - 0.15) / 0.45, 0, 1))  # 0.15 -> no colour, 0.60+ -> full colour
        if strength < 0.05:
            return html.escape(word)
        return (f'<span title="similarity {sim:.2f}" style="background: rgba(57,135,229,{0.1 + 0.45*strength:.2f});'
                f'padding:0 2px">{html.escape(word)}</span>')
    return decorate_words(text, decorate)


def result_card(rank, doc_index, score, color, query, method):
    doc = docs.iloc[doc_index]
    keywords = se.shared_keywords(query, doc["text"], vectorizer)
    if method == "Keyword (TF-IDF)" or not show_heat:
        body = highlight_keywords(doc["text"], keywords)
    else:
        body = highlight_meaning(doc["text"], se.word_meaning_scores(query, doc["text"], model))
    is_correct = doc["id"] in expected_answers.get(query.strip().lower(), set())
    tick = " &middot; <b style='color:#199e70'>expected answer</b>" if is_correct else ""
    why = ", ".join(keywords) if keywords else "none - matched on meaning only"
    cat_color = CATEGORY_COLORS.get(doc["category"], "#3987e5")
    st.markdown(f"""
    <div class="card {'correct' if is_correct else ''}">
      <div class="card-top"><span class="rank">#{rank}</span>
        <span class="badge" style="--c:{cat_color}">{html.escape(doc['category'])}</span>{tick}
        <span class="score">{score:.3f}</span></div>
      <div class="title">{html.escape(doc['title'])}</div>
      <div class="bar"><div style="width:{max(2, min(score, 1)*100):.0f}%; background:{color}"></div></div>
      <div class="text">{body}</div>
      <div class="why">Shared keywords: {html.escape(why)}</div>
    </div>""", unsafe_allow_html=True)


def chart_layout(fig, height=420, title=None):
    fig.update_layout(height=height, title=title, paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(color="#333"), margin=dict(l=20, r=20, t=50 if title else 20, b=20),
                      legend=dict(bgcolor="rgba(0,0,0,0)"),
                      hoverlabel=dict(bgcolor="white"))
    fig.update_xaxes(gridcolor="#eee", zerolinecolor="#ddd")
    fig.update_yaxes(gridcolor="#eee", zerolinecolor="#ddd")
    return fig


# Query box (shared by all tabs)
EXAMPLES = [
    "why can't I remember anything after pulling an all-nighter",
    "how can I keep hackers out of my accounts",
    "what killed the dinosaurs",
    "how to stop crying while chopping vegetables",
    "I feel like vomiting on the ferry",
    "should I invest everything in one company",
    "solar panels electricity",
]
if "query" not in st.session_state:
    st.session_state.query = EXAMPLES[0]

picked = st.pills("Examples:", EXAMPLES, selection_mode="single", key="example_pick")
if picked and picked != st.session_state.get("last_pick"):
    st.session_state.query = picked
    st.session_state.last_pick = picked

query = st.text_input("Your question", key="query",
                      placeholder="e.g. my phone dies too quickly").strip()

if not query:
    st.info("Type a question above (or click an example) to start.")
    st.stop()

# --- The core NLP: score every document with every method ---
kw = se.keyword_scores(query, vectorizer, tfidf_matrix)
sem = se.semantic_scores(query, model, embeddings)
hyb = se.hybrid_scores(kw, sem, alpha)
all_scores = {"Keyword (TF-IDF)": kw, "Semantic (SBERT)": sem, "Hybrid": hyb}

tab_battle, tab_galaxy, tab_lab, tab_hood, tab_eval = st.tabs(
    ["Search", "Document map", "Compare sentences", "How it works", "Evaluation"])

# ---- tab 1: Search battle ----
with tab_battle:
    best_kw, best_sem = int(np.argmax(kw)), int(np.argmax(sem))
    if sem.max() < 0.25:  # even the best meaning match is weak
        verdict = (f"<b>No strong match.</b> The best meaning score is only {sem.max():.2f}, so the collection "
                   f"probably does not contain an answer to this query. Semantic search always returns its "
                   f"closest documents, even when none of them is a real match.")
    elif kw.max() == 0:
        verdict = (f"<b>Keyword search is blind here.</b> After removing stop words, none of your query's words "
                   f"appear in any document. Semantic search still understood you and picked "
                   f"<b>“{html.escape(docs.iloc[best_sem]['title'])}”</b>.")
    elif best_kw != best_sem:
        verdict = (f"<b>The methods disagree.</b> Keyword search prefers <b>“{html.escape(docs.iloc[best_kw]['title'])}”</b> "
                   f"(shared words), while semantic search prefers <b>“{html.escape(docs.iloc[best_sem]['title'])}”</b> (shared meaning).")
    else:
        verdict = (f"<b>Both methods agree</b> on <b>“{html.escape(docs.iloc[best_sem]['title'])}”</b>. "
                   f"Your query shares both words and meaning with it.")
    st.markdown(f'<div class="verdict">{verdict}</div>', unsafe_allow_html=True)

    subtitles = {"Keyword (TF-IDF)": "Matches exact words · yellow = shared keyword",
                 "Semantic (SBERT)": "Matches meaning · blue = word close to query meaning",
                 "Hybrid": f"{alpha:.2f} × meaning + {1-alpha:.2f} × keywords"}
    columns = st.columns(3)
    for column, (method, scores) in zip(columns, all_scores.items()):
        with column:
            color = METHOD_COLORS[method]
            st.markdown(f'<div class="col-head" style="--c:{color}">{method}<small>{subtitles[method]}</small></div>',
                        unsafe_allow_html=True)
            if method == "Keyword (TF-IDF)" and scores.max() == 0:
                st.markdown('<div class="card"><div class="title">No results</div>'
                            '<div class="text">Every document scored 0. TF-IDF cannot see synonyms or paraphrases.</div></div>',
                            unsafe_allow_html=True)
                continue
            shown = se.top_k(scores, top_n)
            if method == "Keyword (TF-IDF)":
                shown = [i for i in shown if scores[i] > 0]  # a score of 0 means "no shared word", not a match
            for rank, doc_index in enumerate(shown, start=1):
                result_card(rank, doc_index, float(scores[doc_index]), color, query, method)

# ---- tab 2: Meaning galaxy (2-D maps of both vector spaces) ----
with tab_galaxy:
    st.markdown("Every dot is a document. Documents that are **close together are similar** according to that method. "
                "The star is your query, and lines connect it to that method's top-3 results.")
    query_emb = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    sem_xy, sem_q, sem_var = se.project_semantic_space(embeddings, query_emb)
    kw_xy, kw_q, kw_var = se.project_keyword_space(tfidf_matrix, vectorizer.transform([query]))

    def galaxy_figure(doc_xy, query_xy, scores, title, color):
        fig = go.Figure()
        for category in docs["category"].unique():
            mask = (docs["category"] == category).to_numpy()
            fig.add_trace(go.Scatter(
                x=doc_xy[mask, 0], y=doc_xy[mask, 1], mode="markers", name=category,
                marker=dict(size=11, color=CATEGORY_COLORS.get(category), symbol=CATEGORY_SYMBOLS.get(category),
                            line=dict(width=1, color="white")),
                text=docs.loc[mask, "title"], customdata=scores[mask],
                hovertemplate="<b>%{text}</b><br>" + category + "<br>score %{customdata:.3f}<extra></extra>"))
        if scores.max() > 0:
            for i in se.top_k(scores, 3):
                fig.add_trace(go.Scatter(x=[query_xy[0], doc_xy[i, 0]], y=[query_xy[1], doc_xy[i, 1]], mode="lines",
                                         line=dict(color=color, width=2, dash="dot"), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[query_xy[0]], y=[query_xy[1]], mode="markers+text", name="Your query",
                                 marker=dict(size=24, symbol="star", color="#222", line=dict(width=1, color=color)),
                                 text=["  query"], textposition="middle right", hovertext=[query], hoverinfo="text"))
        fig.update_xaxes(showticklabels=False, title=None)
        fig.update_yaxes(showticklabels=False, title=None)
        return chart_layout(fig, 520, title)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(galaxy_figure(kw_xy, kw_q, kw, "Keyword space (TF-IDF → SVD 2-D)", METHOD_COLORS["Keyword (TF-IDF)"]),
                        width="stretch")
        st.caption(f"These 2 axes keep only {kw_var:.0%} of the TF-IDF variance. Most documents share almost no "
                   "words with each other, so they pile up near the centre with topics mixed together.")
    with right:
        st.plotly_chart(galaxy_figure(sem_xy, sem_q, sem, "Meaning space (Sentence-BERT → PCA 2-D)", METHOD_COLORS["Semantic (SBERT)"]),
                        width="stretch")
        st.caption(f"These 2 axes keep {sem_var:.0%} of the embedding variance. Documents spread out more and "
                   "same-topic documents tend to sit nearer each other, although the model was never given topic labels.")

    if docs["category"].nunique() > 1:
        sil_kw = se.category_separation(tfidf_matrix.toarray(), docs["category"])
        sil_sem = se.category_separation(embeddings, docs["category"])
        c1, c2 = st.columns(2)
        c1.metric("Topic clustering – TF-IDF (silhouette, full vectors)", f"{sil_kw:.3f}")
        c2.metric("Topic clustering – SBERT (silhouette, full vectors)", f"{sil_sem:.3f}",
                  delta=f"{sil_sem - sil_kw:+.3f} vs TF-IDF")
        st.caption("Silhouette score ranges from −1 to 1. Higher means documents of the same category sit closer "
                   "to each other than to other categories. Computed on the full vectors, not the 2-D picture.")

# ---- tab 3: Similarity lab (compare any two sentences) ----
with tab_lab:
    st.markdown("Compare **any two sentences**. TF-IDF only counts shared words; Sentence-BERT compares meaning.")
    # preset name: (sentence A, sentence B, what to notice)
    PAIRS = {
        "Paraphrase": ("The movie was fantastic", "I really enjoyed the film",
                       "No shared words, so TF-IDF says 0. Sentence-BERT sees the same meaning."),
        "Synonyms": ("I deposited money at the bank", "I put cash into my savings account",
                     "money ↔ cash and bank ↔ savings account are synonyms, which TF-IDF cannot see."),
        "Same word, different meaning": ("Photosynthesis happens in leaves", "The train leaves at nine",
                                         "TF-IDF is fooled by the shared word 'leaves' and scores higher than Sentence-BERT, "
                                         "which uses context to tell the plant part from the verb."),
        "Negation": ("The food was not good", "The food was good",
                     "TF-IDF removes 'not' as a stop word, so it calls these sentences identical (1.00). "
                     "Sentence-BERT keeps 'not' and gives a lower score, though still high."),
        "Word order (limitation)": ("The dog bit the man", "The man bit the dog",
                                    "Honest limitation: TF-IDF ignores word order completely (1.00), and even "
                                    "Sentence-BERT scores these as near-identical."),
        "Question vs answer": ("How do I reset my password?", "Click 'forgot password' to get a reset link by email",
                               "Both methods see a link here. Sentence-BERT is more confident."),
    }
    pair_name = st.radio("Preset pairs", list(PAIRS), horizontal=True)
    col_a, col_b = st.columns(2)
    sentence_a = col_a.text_area("Sentence A", PAIRS[pair_name][0], height=80, key=f"a_{pair_name}")
    sentence_b = col_b.text_area("Sentence B", PAIRS[pair_name][1], height=80, key=f"b_{pair_name}")
    st.info(f"**What to notice:** {PAIRS[pair_name][2]}")

    if sentence_a.strip() and sentence_b.strip():
        kw_sim, sem_sim, shared = se.compare_two_sentences(sentence_a, sentence_b, model)
        m1, m2, m3 = st.columns(3)
        m1.markdown(f'<div class="metric-card" style="--c:{METHOD_COLORS["Keyword (TF-IDF)"]}"><div class="big">{kw_sim:.2f}</div>'
                    f'<div class="big-label">TF-IDF cosine (word overlap)</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card" style="--c:{METHOD_COLORS["Semantic (SBERT)"]}"><div class="big">{sem_sim:.2f}</div>'
                    f'<div class="big-label">Sentence-BERT cosine (meaning)</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-card" style="--c:#999"><div class="big" style="font-size:1.3rem;padding:10px 0">'
                    f'{html.escape(", ".join(shared)) if shared else "—"}</div>'
                    f'<div class="big-label">Shared content words</div></div>', unsafe_allow_html=True)

        words_a, words_b, matrix = se.word_similarity_matrix(sentence_a, sentence_b, model)
        if words_a and words_b:
            fig = go.Figure(go.Heatmap(z=matrix, x=words_b, y=words_a, zmin=0, zmax=1,
                                       colorscale="Blues",
                                       texttemplate="%{z:.2f}",
                                       hovertemplate="%{y} ↔ %{x}: %{z:.2f}<extra></extra>"))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(chart_layout(fig, 120 + 45 * len(words_a), "Word-to-word meaning similarity (A rows × B columns)"),
                            width="stretch")
            st.caption("Each cell is the cosine similarity of two individual word embeddings (stop words removed). "
                       "TF-IDF only gives credit when the two words are exactly the same. "
                       "Tip: compare scores within one method. Sentence-BERT rarely gives exactly 0, "
                       "even for unrelated sentences.")

# ---- tab 4: Under the hood (pipeline for the current query) ----
with tab_hood:
    st.markdown(f"Here is exactly what each method does with your query: **“{html.escape(query)}”**")
    left, right = st.columns(2)

    with left:
        st.markdown(f'<div class="col-head" style="--c:{METHOD_COLORS["Keyword (TF-IDF)"]}">Keyword pipeline (TF-IDF)</div>',
                    unsafe_allow_html=True)
        steps = se.preprocessing_steps(query, vectorizer)
        st.markdown('<div class="step">① Lower-case</div>', unsafe_allow_html=True)
        st.code(steps["lowercased"], language=None)
        st.markdown('<div class="step">② Tokenize → ③ remove stop words (crossed out)</div>', unsafe_allow_html=True)
        chips = "".join(f'<span class="chip {"gone" if t in steps["removed_stopwords"] else ""}">{html.escape(t)}</span>'
                        for t in steps["tokens"])
        st.markdown(chips or "—", unsafe_allow_html=True)
        st.markdown('<div class="step">④ Look up words in the vocabulary</div>', unsafe_allow_html=True)
        st.markdown(f"Known words: **{', '.join(steps['in_vocabulary']) or 'none'}**  \n"
                    f"Unknown words (ignored!): **{', '.join(steps['unknown_words']) or 'none'}**  \n"
                    f"Vocabulary size: {len(vectorizer.vocabulary_)} words → the query becomes a "
                    f"{len(vectorizer.vocabulary_)}-dimensional sparse vector.")
        st.markdown('<div class="step">⑤ TF-IDF weights of the query</div>', unsafe_allow_html=True)
        if steps["tfidf_weights"]:
            weights = steps["tfidf_weights"]
            fig = go.Figure(go.Bar(x=list(weights.values()), y=list(weights.keys()), orientation="h",
                                   marker=dict(color=METHOD_COLORS["Keyword (TF-IDF)"], cornerradius=4),
                                   hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(chart_layout(fig, 90 + 32 * len(weights)), width="stretch")
            st.caption("Rare words get higher weight (high IDF); words found in many documents get lower weight.")
        else:
            st.warning("The query vector is all zeros, so cosine similarity is 0 for every document.")

    with right:
        st.markdown(f'<div class="col-head" style="--c:{METHOD_COLORS["Semantic (SBERT)"]}">Semantic pipeline (Sentence-BERT)</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="step">① WordPiece sub-word tokenization</div>', unsafe_allow_html=True)
        pieces = ["[CLS]"] + model.tokenizer.tokenize(query) + ["[SEP]"]
        st.markdown("".join(f'<span class="chip sub">{html.escape(p)}</span>' for p in pieces), unsafe_allow_html=True)
        st.caption("Pieces starting with ## are parts of a longer word, so the model never has an 'unknown word'. "
                   "Stop words are kept, because words like 'not' change meaning.")
        st.markdown('<div class="step">② 6 transformer layers with self-attention</div>', unsafe_allow_html=True)
        st.markdown("Each token looks at every other token and updates its vector based on context "
                    "(this is how *bank* near *river* differs from *bank* near *money*).")
        st.markdown('<div class="step">③ Mean pooling → one 384-number sentence vector</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Heatmap(z=[query_emb], colorscale=[[0, "#d95926"], [0.5, "#ffffff"], [1, "#3987e5"]],
                                   zmid=0, showscale=False, hovertemplate="dimension %{x}: %{z:.3f}<extra></extra>"))
        fig.update_yaxes(showticklabels=False)
        st.plotly_chart(chart_layout(fig, 130), width="stretch")
        st.caption("Your query as 384 numbers (blue = positive, orange = negative). No single number has a human "
                   "meaning; together they encode the sentence.")
        st.markdown('<div class="step">④ Cosine similarity with every document vector</div>', unsafe_allow_html=True)
        st.latex(r"\cos(q, d) = \frac{q \cdot d}{\lVert q \rVert \, \lVert d \rVert}")
        best = se.top_k(sem, 1)[0]
        st.markdown(f"Highest: **{sem[best]:.3f}** with “{docs.iloc[best]['title']}”. "
                    f"Lowest: **{sem.min():.3f}**.")

# ---- tab 5: Evaluation on the hand-labelled test set ----
with tab_eval:
    st.markdown(f"**Test set:** {len(eval_queries)} natural-language questions written in everyday words, each labelled "
                "with the document(s) that answer it. All three methods search the same 80 built-in documents. "
                "These numbers are computed live, right now.")
    per_query, summary = get_evaluation(alpha)

    metric_cols = st.columns(3)
    for column, method in zip(metric_cols, summary.index):
        row = summary.loc[method]
        column.markdown(f'<div class="metric-card" style="--c:{METHOD_COLORS[method]}"><b>{method}</b>'
                        f'<div class="big" style="margin-top:8px">{row["MRR"]:.2f}</div><div class="big-label">MRR</div>'
                        f'<div style="margin-top:8px;color:#555">P@1 {row["precision@1"]:.2f} · '
                        f'Recall@5 {row["recall@5"]:.2f}</div></div>', unsafe_allow_html=True)

    fig = go.Figure()
    for method in summary.index:
        fig.add_trace(go.Bar(name=method, x=["Precision@1", "Recall@5", "MRR"], y=summary.loc[method].tolist(),
                             marker=dict(color=METHOD_COLORS[method], cornerradius=4),
                             text=[f"{v:.2f}" for v in summary.loc[method]], textposition="outside",
                             hovertemplate=method + "<br>%{x}: %{y:.3f}<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.06, yaxis_range=[0, 1.1])
    st.plotly_chart(chart_layout(fig, 400, "Higher is better"), width="stretch")

    # Alpha sweep: how the hybrid changes from pure keyword (0) to pure meaning (1)
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    sweep_mrr = [get_evaluation(a)[1].loc["Hybrid", "MRR"] for a in alphas]
    fig = go.Figure(go.Scatter(x=alphas, y=sweep_mrr, mode="lines+markers", name="Hybrid MRR",
                               line=dict(color=METHOD_COLORS["Hybrid"], width=2), marker=dict(size=9),
                               hovertemplate="α = %{x:.1f}<br>MRR = %{y:.3f}<extra></extra>"))
    fig.add_vline(x=alpha, line=dict(color="#999", dash="dot"), annotation_text=f"current α = {alpha:.2f}")
    fig.update_xaxes(title="α (0 = pure keyword, 1 = pure meaning)")
    fig.update_yaxes(title="MRR", range=[0, 1.05])
    st.plotly_chart(chart_layout(fig, 340, "Hybrid search: effect of the meaning weight α"), width="stretch")
    st.caption("On this test set, giving more weight to meaning keeps improving results, because most questions "
               "are paraphrases. Keyword weight helps more with exact names, codes or rare terms.")

    with st.expander("What do these metrics mean?"):
        st.markdown("""
- **Precision@1**: share of questions where the very first result is a correct answer.
- **Recall@5**: share of the correct documents that appear somewhere in the top 5.
- **MRR (Mean Reciprocal Rank)**: average of 1 / (position of the first correct answer). Rank 1 → 1.0, rank 2 → 0.5, rank 3 → 0.33, not found → 0.
- For TF-IDF, a document with score 0 (no shared word) is counted as *not found*, because the order among zero-score documents is arbitrary.
""")

    st.markdown("#### Question by question: at what position did each method find the answer?")
    table = per_query.pivot(index="query", columns="method", values="rank_of_answer")[list(summary.index)]
    table = table.reindex([q["query"] for q in eval_queries])
    st.dataframe(table.map(lambda r: "not found" if pd.isna(r) else f"#{int(r)}"), width="stretch", height=420)
    st.caption("Honest note: the documents and questions were written for this demo (see data/). The questions "
               "were deliberately phrased in everyday words, which is the situation semantic search is designed for. "
               "Move the α slider in the sidebar to see how the hybrid score changes.")
