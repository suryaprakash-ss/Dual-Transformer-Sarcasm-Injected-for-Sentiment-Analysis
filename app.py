import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/analyze/"

st.set_page_config(page_title="DTS² Analyzer", layout="centered")
st.title("DTS² - Sarcasm & Sentiment Analyzer")

user_input = st.text_area(
    "Enter text to analyze:",
    height=150,
    placeholder="Type a sarcastic or emotional sentence here..."
)

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter some text first!")
    else:
        with st.spinner("Analyzing..."):
            try:
                result = requests.post(API_URL, json={"text": user_input}).json()

                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    sarcasm = float(result.get("sarcasm_score", 0))
                    emotion = result.get("emotion", "Unknown")
                    sentiment = result.get("sentiment", "Unknown")
                    final = result.get("final_sentiment", "Unknown")

                    # Sarcasm Score
                    st.subheader("Sarcasm Score")
                    st.progress(min(sarcasm, 1.0))
                    st.write(f"**Score:** {sarcasm:.2f}")

                    # Emotion & Base Sentiment
                    st.subheader("Emotion & Base Sentiment")
                    st.info(f"**Emotion:** {emotion}")
                    st.write(f"**Base Sentiment:** {sentiment}")

                    # Final Sentiment
                    st.subheader("Final Sentiment")
                    if final.lower() == "positive":
                        st.success(f"{final}")
                    elif final.lower() == "negative":
                        st.error(f"{final}")
                    else:
                        st.warning(f"{final}")

            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to backend: {e}")

st.markdown("---")
st.caption("© 2025 DTS² Research Prototype | Internal Demo Only")
