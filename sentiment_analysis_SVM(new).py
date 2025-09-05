
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
    # Basic validation
    required = {"text", "label"}
    if df is None or df.empty or not required.issubset(df.columns):
        return None, None

    # Normalize labels (robust to variants)
    label_map = {
        "neg": "negative", "negative": "negative", "NEGATIVE": "negative",
        "neu": "neutral",  "neutral": "neutral",   "NEUTRAL": "neutral",
        "pos": "positive", "positive": "positive", "POSITIVE": "positive",
    }
    df = df.copy()
    df["label"] = df["label"].astype(str).map(lambda x: label_map.get(x.strip(), x.strip()))

    # Train/validation split (stratify only if we truly have >1 class)
    strat = df["label"] if df["label"].nunique() > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        df["text"].astype(str),
        df["label"].astype(str),
        test_size=0.2,
        random_state=42,
        stratify=strat
    )

    # --- Vectorizer: a bit stronger for small datasets ---
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,              # keep rare tokens; dataset is small
        max_df=0.95,
        stop_words="english",
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
    )

    # --- Base classifier ---
    base_svm = LinearSVC(C=1.0, class_weight="balanced")

    # --- Decide on calibration safely ---
    # If any class in TRAIN set has too few examples, reduce cv. If still too low, skip calibration.
    vc = y_train.value_counts()
    min_class = int(vc.min()) if not vc.empty else 0

    # We need at least 2 samples per class for cv>=2
    if min_class >= 3:
        safe_cv = min(5, min_class)  # e.g., if min_class=3 → cv=3
        clf = CalibratedClassifierCV(estimator=base_svm, cv=safe_cv, method="sigmoid")
        model = Pipeline([("tfidf", tfidf), ("clf", clf)])
    elif min_class == 2:
        clf = CalibratedClassifierCV(estimator=base_svm, cv=2, method="sigmoid")
        model = Pipeline([("tfidf", tfidf), ("clf", clf)])
    else:
        # Not enough data to calibrate reliably; use plain LinearSVC (no predict_proba)
        model = Pipeline([("tfidf", tfidf), ("clf", base_svm)])

    # Train
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
        "labels": sorted(df["label"].unique().tolist())
    }
    return model, metrics
