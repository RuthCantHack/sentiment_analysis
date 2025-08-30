
import streamlit as st
st.set_page_config(page_title="Simple Chat", page_icon="💬")  # must be first Streamlit call

import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure VADER lexicon is available (downloads once)
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()

# --- Settings (change if you want) ---
WARN_AT = 3   # warn on 3rd repeat
BAN_AT  = 5   # ban on 5th repeat (total)

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []        # list of dicts: {"role","text","sentiment","compound"}
if "counts" not in st.session_state:
    st.session_state.counts = {}          # normalized_text -> count
if "banned" not in st.session_state:
    st.session_state.banned = False

def normalize(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t

def label_from_compound(c: float) -> str:
    if c >= 0.05:
        return "Positive"
    elif c <= -0.05:
        return "Negative"
    return "Neutral"

st.title("💬 Simple Chat (Sentiment + Repeat Guard)")

# --- Show chat history ---
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["text"])
        if m.get("sentiment") is not None:
            st.caption(f"Sentiment: **{m['sentiment']}** (compound {m['compound']:+.3f})")

# --- Input ---
prompt = st.chat_input("Type a message…", disabled=st.session_state.banned)

if prompt:
    # Sentiment
    scores = sia.polarity_scores(prompt)
    comp = scores["compound"]
    sentiment = label_from_compound(comp)

    # Store user's message
    st.session_state.messages.append({
        "role": "user",
        "text": prompt,
        "sentiment": sentiment,
        "compound": comp
    })

    # Repeat check
    key = normalize(prompt)
    st.session_state.counts[key] = st.session_state.counts.get(key, 0) + 1
    c = st.session_state.counts[key]

    if c == WARN_AT:
        st.session_state.messages.append({
            "role": "assistant",
            "text": f"⚠️ Warning: you've repeated the same message {c} times."
        })

    if c >= BAN_AT and not st.session_state.banned:
        st.session_state.banned = True
        st.session_state.messages.append({
            "role": "assistant",
            "text": f"🚫 You are banned for repeating the same message {c} times."
        })

    st.rerun()
