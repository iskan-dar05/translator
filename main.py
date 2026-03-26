import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences 
import numpy as np

MODEL_PATH="nmt/model.keras"
TOKENIZER_PATH="nmt/tokenizer.pkl"
MAX_LEN = 40



with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)


model = load_model(MODEL_PATH)

LANG_TOKEN = {
    ("English", "Fransh"): "<en> <to_fr>",
    ("English", "Spanish"): "<en> <to_es>",
    ("Fransh", "English"): "<fr> <to_en>",
    ("Spanish", "English"): "<es> <to_en>",
    ("Fransh", "Spanish"): "<fr> <to_es>",
    ("Spanish", "Fransh"): "<es> <to_fr>",
}


def translate(text, option1, option2):
    key = (option1, option2)

    if key not in LANG_TOKEN:
        return "Translation not supported"

    seq_text = LANG_TOKEN[key] + " " + text.lower().strip()

    encoded_seq = tokenizer.texts_to_sequences([seq_text])
    encoded_seq = pad_sequences(encoded_seq, maxlen=MAX_LEN, padding="post")


    sos_id = tokenizer.word_index["<sos>"]
    eos_id = tokenizer.word_index["<eos>"]

    decoder_input = pad_sequences([[sos_id]], maxlen=MAX_LEN, padding="post")

    result = []

    result_ids = []

    for _ in range(MAX_LEN):
        output = model.predict([encoded_seq, decoder_input])
        t = len(result_ids)  # current timestep
        token_id = np.argmax(output[0, t])

        if token_id == eos_id:
            break

        result_ids.append(token_id)

        word = tokenizer.index_word[token_id]
        result.append(word)

        decoder_input = pad_sequences(
            [[sos_id] + result_ids],
            maxlen=MAX_LEN,
            padding="post"
        )

    return " ".join(result)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Multi-Lang LSTM Translator",
    page_icon="🌍",
    layout="wide"
)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🌍 LSTM Translator")

    st.markdown("""
    ### 📌 Model Overview
    This application uses a **Sequence-to-Sequence LSTM model**
    trained for **multilingual translation**.

    **Supported languages:**
    - 🇬🇧 English
    - 🇫🇷 French
    - 🇪🇸 Spanish

    ⚠️ This model **does not support other languages**.

    ---
    ### 🧠 How it works
    - Encoder LSTM reads the input sentence
    - Decoder LSTM generates the translated sentence
    - Word embeddings represent tokens
    - Teacher forcing was used during training

    ---
    ### 🚀 Notes
    - Auto-translation (no button)
    - Output is read-only
    - Designed for learning & demo purposes
    """)

# ---------------- TITLE ----------------
st.markdown(
    "<h1 style='text-align: center;'>🌐 Multilingual LSTM Translator</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; color: gray;'>English ↔ French ↔ Spanish</p>",
    unsafe_allow_html=True
)

st.divider()

# ---------------- LAYOUT ----------------
col1, col2 = st.columns(2)

with col1:
    option1 = st.selectbox(
        "From",
        ("English", "Fransh", "Spanish"),
        key="lang_1"
    )

    txt = st.text_area(
        "Enter text",
        "",
        height=220,
        key="input_text"
    )

with col2:
    option2 = st.selectbox(
        "To",
        ("English", "Fransh", "Spanish"),
        key="lang_2"
    )

# ---------------- SESSION STATE ----------------
if "last_input" not in st.session_state:
    st.session_state.last_input = ""
    st.session_state.translation = ""

# ---------------- AUTO TRANSLATE ----------------
if txt != st.session_state.last_input:
    st.session_state.last_input = txt

    if txt and option1 != option2:
        st.session_state.translation = translate(txt, option1, option2)
    else:
        st.session_state.translation = txt

# ---------------- OUTPUT ----------------
with col2:
    st.markdown("### Translation")
    st.text_area(
        "",
        st.session_state.translation,
        height=220,
        disabled=True
    )

# ---------------- FOOTER ----------------
st.divider()
st.markdown(
    "<p style='text-align:center; color: gray;'>Built with ❤️ using Streamlit & LSTM Seq2Seq</p>",
    unsafe_allow_html=True
)
