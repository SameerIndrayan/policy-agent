import streamlit as st
import requests

st.set_page_config(page_title="Policy Agent", layout="wide")
st.title("Policy Agent")

api_url = st.text_input("API URL", "http://127.0.0.1:8000")
policy = st.text_area("Policy (optional)", height=120, placeholder="Paste policy here…")
text = st.text_area("Input to evaluate", height=160, placeholder="Paste text to check…")

if st.button("Evaluate", use_container_width=True):
    if not text.strip():
        st.warning("Enter some text to evaluate.")
    else:
        with st.spinner("Evaluating…"):
            try:
                r = requests.post(
                    f"{api_url}/evaluate",
                    json={"text": text, "policy": policy or None},
                    timeout=30,
                )
                r.raise_for_status()
                data = r.json()
                st.metric("Allowed", "✅ Yes" if data.get("allowed") else "❌ No")
                st.write("**Reason**")
                st.info(data.get("reason") or "—")
            except Exception as e:
                st.error(f"HTTP error: {e}")
