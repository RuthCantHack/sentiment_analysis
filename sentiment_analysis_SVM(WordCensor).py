import re
from pathlib import Path
import streamlit as st
import pandas as pd

# ML / NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# ---------------- Page title ----------------
st.set_page_config(page_title="Sentiment Analysis (SVM, no autocorrect)", page_icon="🤖")

# ---------------- Cleaning ----------------
def simple_clean(s: str) -> str:
    """Collapse whitespace and reduce repeated characters."""
    s = re.sub(r"\s+", " ", s or "").strip()
    s = re.sub(r"(.)\1{2,}", r"\1\1", s)  # "soooo" -> "soo"
    return s

# ---------------- Dataset loader ----------------
def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "label" in df.columns:
        label_map = {
            "neg":"negative","negative":"negative","NEGATIVE":"negative",
            "neu":"neutral","neutral":"neutral","NEUTRAL":"neutral",
            "pos":"positive","positive":"positive","POSITIVE":"positive",
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
    return pd.DataFrame(columns=["text","label"]), None

TARGET_WORDS = ["fuck", "shit", "bitch", "stupid", "idiot", "bastard"]

LEET_MAP = {"a":"[a@]", "i":"[i1!]", "o":"[o0]", "e":"[e3]", "s":"[s5$]", "t":"[t7]"}
def _leetify(word: str) -> str:
    return "".join(LEET_MAP.get(ch, re.escape(ch)) for ch in word.lower())

CENSOR_PATTERNS = [re.compile(rf"\b{_leetify(w)}\b", re.IGNORECASE) for w in TARGET_WORDS]

def censor_ui_only(text: str, patterns, track: bool = False):
    hits = []
    def _rep(m):
        token = m.group(0); hits.append(token)
        return "*" * len(token)
    out = text
    for patt in patterns:
        out = patt.sub(_rep if track else (lambda m: "*" * len(m.group(0))), out)
    return (out, hits) if track else out

# ---------------- Model builder (TF-IDF + SVM) ----------------
@st.cache_resource(show_spinner=True)
def build_model(df: pd.DataFrame):
    required = {"text","label"}
    if df is None or df.empty or not required.issubset(df.columns):
        return None, None

    model_text = df["text"].astype(str).apply(simple_clean)

    strat = df["label"] if df["label"].nunique() > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        model_text,
        df["label"].astype(str),
        test_size=0.2,
        random_state=42,
        stratify=strat,
    )

    tfidf = TfidfVectorizer(
        ngram_range=(1,2),
        min_df=1, max_df=0.95,
        stop_words="english",
        sublinear_tf=True, lowercase=True, strip_accents="unicode",
    )

    base_svm = LinearSVC(C=1.0, class_weight="balanced")


    vc = y_train.value_counts()
    min_class = int(vc.min()) if not vc.empty else 0
    if min_class >= 3:
        clf = CalibratedClassifierCV(estimator=base_svm, cv=min(5, min_class), method="sigmoid")
        model = Pipeline([("tfidf", tfidf), ("clf", clf)])
    elif min_class == 2:
        clf = CalibratedClassifierCV(estimator=base_svm, cv=2, method="sigmoid")
        model = Pipeline([("tfidf", tfidf), ("clf", clf)])
    else:
        model = Pipeline([("tfidf", tfidf), ("clf", base_svm)])  # no probabilities

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "f1_macro": f1_score(y_val, y_pred, average="macro"),
        "report": classification_report(y_val, y_pred, output_dict=False),
        "confusion_matrix": confusion_matrix(y_val, y_pred),
        "labels": sorted(df["label"].unique().tolist()),
        "train_counts": vc.to_dict(),
    }
    return model, metrics

# ---------------- Load, build & UI ----------------
df, dataset_path = load_dataset()
model, metrics = build_model(df)

st.title("🤖 Review Sentiment (SVM, dataset-trained)")
if dataset_path:
    st.caption(f"Training data: `{dataset_path}`  |  rows: {len(df)}  |  class counts: {df['label'].value_counts().to_dict()}")
else:
    st.error("Could not find the dataset CSV. Place it next to this app.")
    st.stop()

if metrics:
    c1, c2 = st.columns(2)
    with c1: st.metric("Accuracy (val)", f"{metrics['accuracy']:.3f}")
    with c2: st.metric("F1 (macro, val)", f"{metrics['f1_macro']:.3f}")
    with st.expander("Classification report"): st.text(metrics["report"])
    with st.expander("Confusion matrix"):
        cm_df = pd.DataFrame(metrics["confusion_matrix"], index=metrics["labels"], columns=metrics["labels"])
        st.dataframe(cm_df, use_container_width=True)

st.divider()
st.subheader("Try a review")

review = st.chat_input("Write your review here and press Enter…", key="review_input_box")

if review:
    # ---------- DISPLAY BRANCH (censored) ----------
    review_clean = simple_clean(review)
    corrected_display, hits = censor_ui_only(review_clean, CENSOR_PATTERNS, track=True)

    # ---------- MODEL BRANCH (uncensored) ----------
    text_for_model = simple_clean(review)

    # Show original in chat style
    with st.chat_message("user"):
        st.write(review)

    # Show corrected (censored) text to user
    st.subheader("Corrected review (display)")
    st.write(corrected_display)

    # List of censored tokens
    if hits:
        st.caption("Censored tokens: " + ", ".join(hits))

    # Predict on uncensored text
    if model is None:
        st.error("Model not available (dataset missing or invalid).")
    else:
        pred = model.predict([text_for_model])[0]
        try:
            proba = model.predict_proba([text_for_model])[0]
            labels = model.classes_.tolist()
            prob_df = pd.DataFrame({"label": labels, "probability": proba}).sort_values("probability", ascending=False)
            st.subheader("Sentiment result")
            st.success(f"Prediction: **{pred}**")
            st.dataframe(prob_df, use_container_width=True)
        except Exception:
            st.subheader("Sentiment result")
            st.success(f"Prediction: **{pred}**")
