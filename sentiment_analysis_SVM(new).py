
import re
from pathlib import Path
import streamlit as st
import pandas as pd
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# ---- Page config ----
st.set_page_config(page_title="Review Sentiment (SVM, dataset-trained)", page_icon="🤖")

# ---- Autocorrect Utilities ----
spell = SpellChecker(language="en")
WORD_RE = re.compile(r"[A-Za-z']+|[^A-Za-z']")

def _preserve_case(src: str, dst: str) -> str:
    if src.isupper():
        return dst.upper()
    if src[:1].isupper() and src[1:].islower():
        return dst.capitalize()
    return dst

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

# ---- Data loader ----
@st.cache_data(show_spinner=False)
def load_dataset():
    # Try to find sentiment_dataset.csv next to this script; fall back to working dir.
    here = Path(__file__).parent
    candidates = [
        here / "sentiment_dataset.csv",
        Path("sentiment_dataset.csv"),
        Path("/mount/src/sentiment_analysis/sentiment_dataset.csv"),  # Streamlit Cloud typical path
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            return df, str(p)
    # If not found, return empty df
    return pd.DataFrame(columns=["text","label"]), None

df, dataset_path = load_dataset()

# ---- Model builder (cached) ----
@st.cache_resource(show_spinner=True)
def build_model(df: pd.DataFrame):
    if df.empty or not set(["text","label"]).issubset(df.columns):
        return None, None

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        df["text"].astype(str),
        df["label"].astype(str),
        test_size=0.2,
        random_state=42,
        stratify=df["label"].astype(str) if df["label"].nunique() > 1 else None
    )

    # Stronger text features + class balancing + calibrated probabilities
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1,2),         # unigrams + bigrams
            min_df=2,                  # ignore very rare tokens
            max_df=0.9,                # drop super common tokens
            sublinear_tf=True,         # log(1 + tf)
            lowercase=True,
            strip_accents="unicode",
        )),
        ("clf", CalibratedClassifierCV(
            base_estimator=LinearSVC(class_weight="balanced"),
            cv=5
        ))
    ])
    model.fit(X_train, y_train)

    # Evaluate
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
        "labels": sorted(df["label"].astype(str).unique().tolist())
    }
    return model, metrics

model, metrics = build_model(df)

# ---- UI ----
st.title("🤖 Review Autocorrect + Sentiment (SVM, dataset-trained)")
if dataset_path:
    st.caption(f"Training data: `{dataset_path}`  |  {len(df)} rows")
else:
    st.error("Could not find `sentiment_dataset.csv`. Place it next to the app or in the working directory.")
    st.stop()

# Show some metrics
if metrics:
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy (val)", f"{metrics['accuracy']:.3f}")
    with col2:
        st.metric("F1 (macro, val)", f"{metrics['f1_macro']:.3f}")
    with st.expander("Classification report"):
        st.text(metrics["report"])
    with st.expander("Confusion matrix"):
        import numpy as np
        import pandas as pd
        labels = metrics["labels"]
        cm_df = pd.DataFrame(metrics["confusion_matrix"], index=labels, columns=labels)
        st.dataframe(cm_df, use_container_width=True)

st.divider()
st.subheader("Try a review")

review = st.chat_input("Write your review here and press Enter…", key="review_input_box")

if review:
    corrected, changes = autocorrect_text(review)
    with st.chat_message("user"):
        st.write(review)

    st.subheader("Corrected review")
    st.write(corrected)
    st.subheader("Autocorrect details")
    st.markdown(highlight_changes(changes))

    if model is None:
        st.error("Model not available (dataset missing or invalid).")
    else:
        pred = model.predict([corrected])[0]
        # show probabilities if available
        proba = None
        try:
            import numpy as np
            proba = model.predict_proba([corrected])[0]
            labels = model.classes_.tolist()
            prob_df = pd.DataFrame({"label": labels, "probability": proba})
            prob_df = prob_df.sort_values("probability", ascending=False)
            st.subheader("Sentiment result")
            st.success(f"Prediction: **{pred}**")
            st.dataframe(prob_df, use_container_width=True)
        except Exception:
            st.subheader("Sentiment result")
            st.success(f"Prediction: **{pred}**")
