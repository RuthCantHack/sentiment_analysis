
import re
import streamlit as st
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline

# ---- Page config (declare exactly once) ----
st.set_page_config(page_title="Review Sentiment (SVM)", page_icon="🤖")

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
    tokens = WORD_RE.findall(text)
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

# ---- Train SVM once (cached) ----
@st.cache_resource
def build_model():
    train_texts = [
        "I love this product, it's amazing",
        "This is the best service ever",
        "Absolutely fantastic experience",
        "I hate this, it was terrible",
        "Worst purchase I have made",
        "Really bad and disappointing",
        "It was okay, nothing special",
        "Not bad, pretty average overall",
        "The experience was fine",
    ]
    train_labels = [
        "Positive", "Positive", "Positive",
        "Negative", "Negative", "Negative",
        "Neutral", "Neutral", "Neutral"
    ]
    model = make_pipeline(TfidfVectorizer(), LinearSVC())
    model.fit(train_texts, train_labels)
    return model

svm_model = build_model()

def classify_sentiment(text: str):
    return svm_model.predict([text])[0]

# ---- UI ----
st.title("🤖 Review Autocorrect + Sentiment (SVM)")
st.caption("Type a review, I'll fix spelling mistakes and classify it using an SVM model.")

# Unique key prevents duplicate element id errors
review = st.chat_input("Write your review here and press Enter…", key="review_input_box")

if review:
    corrected, changes = autocorrect_text(review)
    label = classify_sentiment(corrected)

    # show the user message in chat style
    with st.chat_message("user"):
        st.write(review)

    st.subheader("Corrected review")
    st.write(corrected)

    st.subheader("Autocorrect details")
    st.markdown(highlight_changes(changes))

    st.subheader("Sentiment result")
    st.success(f"Prediction: **{label}**")
