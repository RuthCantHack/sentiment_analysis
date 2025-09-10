# sentiment_analysis_SVM.py  (with swear-word censoring)
import re
from pathlib import Path
import streamlit as st
import pandas as pd

# ML / NLP
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# ---------------- Page config ----------------
st.set_page_config(page_title="Review Autocorrect + Sentiment (SVM)", page_icon="🤖")

# ---------------- Autocorrect + cleaning utilities ----------------
spell = SpellChecker(language="en")
WORD_RE = re.compile(r"[A-Za-z']+|[^A-Za-z']")

def _preserve_case(src: str, dst: str) -> str:
    if src.isupper():
        return dst.upper()
    if src[:1].isupper() and src[1:].islower():
        return dst.capitalize()
    return dst

def simple_clean(s: str) -> str:
    """Collapse whitespace and extreme repeats: 'soooo' -> 'soo'."""
    s = re.sub(r"\s+", " ", s or "").strip()
    s = re.sub(r"(.)\1{2,}", r"\1\1", s)
    return s

# ---------- NEW: censor swear words with '*' ----------
# Expand this list as needed for your project rubric.
SWEAR_WORDS = [
    "shit", "fuck", "bitch", "bastard", "asshole", "dick", "crap"
]

# Optional variants (very light leetspeak handling)
LEET_MAP = {
    "a": "[a@]",
    "i": "[i1!]",
    "o": "[o0]",
    "e": "[e3]",
    "s": "[s5$]",
    "t": "[t7]"
}
def _leetify(word: str) -> str:
    patt = ""
    for ch in word.lower():
        patt += LEET_MAP.get(ch, re.escape(ch))
    return patt

CENSOR_PATTERNS = [
    re.compile(rf"\b{_leetify(w)}\b", flags=re.IGNORECASE) for w in SWEAR_WORDS
]

def censor_swear_words(text: str, track: bool = False):
    """Replace detected swear words with '*'"""
    hits = []

    def _replace(m):
        token = m.group(0)
        hits.append(token)
        return "*" * len(token)

    out = text
    for patt in CENSOR_PATTERNS:
        out = patt.sub(_replace if track else (lambda m: "*" * len(m.group(0))), out)

    if track:
        return out, hits
    return out


def autocorrect_text(text: str):
    tokens = WORD_RE.findall(text or "")
    changed = []
    corrected_tokens = []
    for tok in tokens:
        if tok.isalpha() and len(tok) > 1:
            lower = tok.lower()
            if lower not in spell and lower != "i":
                candidate = spell.correction(lower) or lower
                if candidate != lower:
                    fixed = _preserve_case(tok, candidate)
                    corrected_tokens.append(fixed)
                    changed.append((tok, fixed))
                else:
                    corrected_tokens.append(tok)
            else:
                corrected_tokens.append(tok)
        else:
            corrected_tokens.append(tok)
    return "".join(corrected_tokens), changed

def highlight_changes(changed_pairs):
    if not changed_pairs:
        return "No spelling changes were needed."
    lines = [f"- **{o}** → **{c}**" for o, c in changed_pairs]
    return "Autocorrections:\n" + "\n".join(lines)

# ---------------- Dataset loader (Jupyter-safe) ----------------
def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "label" in df.columns:
        label_map = {
            "neg": "negative", "negative": "negative", "NEGATIVE": "negative",
            "neu": "neutral",  "neutral": "neutral",   "NEUTRAL": "neutral",
            "pos": "positive", "positive": "positive", "POSITIVE": "positive",
        }
        df["label"] = df["label"].astype(str).map(lambda x: label_map.get(x.strip(), x.strip()))
    return df

@st.cache_data(show_spinner=False)
def load_dataset():
    try:
        here = Path(__file__).parent
    except NameError:
        here = Path.cwd()

    candidates = [
        here / "sentiment_dataset(1).csv",
        Path.cwd() / "sentiment_dataset(1).csv",
        Path("sentiment_dataset(1).csv"),
        Path("/mount/src/sentiment_analysis/sentiment_dataset(1).csv"),
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            return normalize_labels(df), str(p)
    return pd.DataFrame(columns=["text", "label"]), None

# ---------------- Model builder (robust calibration) ----------------
@st.cache_resource(show_spinner=True)
def build_model(df: pd.DataFrame):
    required = {"text", "label"}
    if df is None or df.empty or not required.issubset(df.columns):
        return None, None

    strat = df["label"] if df["label"].nunique() > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        df["text"].astype(str),
        df["label"].astype(str),
        test_size=0.2,
        random_state=42,
        stratify=strat,
    )

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        stop_words="english",
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
    )

    base_svm = LinearSVC(C=1.0, class_weight="balanced")

    # Safe calibration depending on smallest class size in TRAIN
    vc = y_train.value_counts()
    min_class = int(vc.min()) if not vc.empty else 0

    if min_class >= 3:
        safe_cv = min(5, min_class)
        clf = CalibratedClassifierCV(estimator=base_svm, cv=safe_cv, method="sigmoid")
        model = Pipeline([("tfidf", tfidf), ("clf", clf)])
    elif min_class == 2:
        clf = CalibratedClassifierCV(estimator=base_svm, cv=2, method="sigmoid")
        model = Pipeline([("tfidf", tfidf), ("clf", clf)])
    else:
        model = Pipeline([("tfidf", tfidf), ("clf", base_svm)])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average="macro")
    report = classification_report(y_val, y_pred, output_dict=False)
    cm = confusion_matrix(y_val, y_pred)

    metrics = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "report": report,
        "confusion_matrix": cm,
        "labels": sorted(df["label"].unique().tolist()),
        "train_counts": vc.to_dict(),
    }
    return model, metrics

# ---------------- Load & UI ----------------
df, dataset_path = load_dataset()
model, metrics = build_model(df)

st.title("🤖 Review Autocorrect + Sentiment (SVM, dataset-trained)")
if dataset_path:
    st.caption(f"Training data: `{dataset_path}`  |  rows: {len(df)}  |  class counts: {df['label'].value_counts().to_dict()}")
else:
    st.error("Could not find the dataset CSV. Place it next to this app.")
    st.stop()

if metrics:
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy (val)", f"{metrics['accuracy']:.3f}")
    with col2:
        st.metric("F1 (macro, val)", f"{metrics['f1_macro']:.3f}")
    with st.expander("Classification report"):
        st.text(metrics["report"])
    with st.expander("Confusion matrix"):
        labels = metrics["labels"]
        cm_df = pd.DataFrame(metrics["confusion_matrix"], index=labels, columns=labels)
        st.dataframe(cm_df, use_container_width=True)

st.divider()
st.subheader("Try a review")

review = st.chat_input("Write your review here and press Enter…", key="review_input_box")

if review:
    # 1) Clean
    review_clean = simple_clean(review)

    # 2) Censor pass 1 (catches raw swears forms before autocorrect)
    review_censored_1, censored_hits_1 = censor_swear_words(review_clean, track=True)

    # 3) Autocorrect
    autocorrected_text, changes = autocorrect_text(review_censored_1)

    # 4) Censor pass 2 
    corrected, censored_hits_2 = censor_swear_words(autocorrected_text, track=True)

    # 5) Show original in chat style
    with st.chat_message("user"):
        st.write(review)

    # 6) Show corrected text & details
    st.subheader("Corrected review")
    st.write(corrected)

    st.subheader("Autocorrect details")
    if changes:
        st.markdown("Autocorrections:\n" + "\n".join([f"- **{o}** → **{c}**" for o, c in changes]))
    else:
        st.markdown("No spelling changes were needed.")

    # 7) Show censoring details
    cens_all = censored_hits_1 + censored_hits_2
    if cens_all:
        st.caption("Censored tokens detected: " + ", ".join({h for h in cens_all}))

    # 8) Predict on the final censored text
    if model is None:
        st.error("Model not available (dataset missing or invalid).")
    else:
        pred = model.predict([corrected])[0]
        try:
            proba = model.predict_proba([corrected])[0]
            labels = model.classes_.tolist()
            prob_df = pd.DataFrame({"label": labels, "probability": proba}).sort_values("probability", ascending=False)
            st.subheader("Sentiment result")
            st.success(f"Prediction: **{pred}**")
            st.dataframe(prob_df, use_container_width=True)
        except Exception:
            st.subheader("Sentiment result")
            st.success(f"Prediction: **{pred}**")
