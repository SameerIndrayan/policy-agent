# ui_frontend.py
import os
import json
import httpx
import streamlit as st

# --- RAG bits (uses your existing rag_setup.py) ---
@st.cache_resource
def _get_query_engine():
    from rag_setup import make_rag_index  # lazy import so UI works even if RAG not ready
    idx = make_rag_index()
    return idx.as_query_engine(similarity_top_k=3)

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Policy Agent & RAG", page_icon="✅", layout="centered")
st.title("Policy Agent & Docs Search")

tab1, tab2 = st.tabs(["🔒 Policy Check", "📚 Ask Docs"])

# ----------------- Tab 1: Policy Check -----------------
with tab1:
    st.subheader("Check text against your policy")
    text = st.text_area("Input text", height=160, placeholder="Paste text to evaluate…")
    policy = st.selectbox(
        "Policy",
        ["default", "forbidden", "lenient", "strict"],
        index=0,
        help="Pick one; 'forbidden' blocks phrases like 'forbidden'.",
    )
    if st.button("Evaluate"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            try:
                payload = {"text": text, "policy": None if policy == "default" else policy}
                r = httpx.post(f"{FASTAPI_URL}/evaluate", json=payload, timeout=30)
                if r.status_code != 200:
                    st.error(f"API error {r.status_code}: {r.text}")
                else:
                    data = r.json()
                    st.markdown(f"**Allowed:** {'✅ Yes' if data.get('allowed') else '❌ No'}")
                    st.markdown(f"**Reason:** {data.get('reason','(no reason)')}")
                    st.code(json.dumps(data, indent=2))
            except Exception as e:
                st.exception(e)

# ----------------- Tab 2: Ask Docs (RAG) -----------------
with tab2:
    st.subheader("Ask your ingested docs")
    q = st.text_input("Question", placeholder="e.g., What role does Weaviate play?")
    colA, colB = st.columns(2)
    with colA:
        if st.button("Ask"):
            if not q.strip():
                st.warning("Please enter a question.")
            else:
                try:
                    qe = _get_query_engine()
                    ans = qe.query(q)
                    st.markdown("**Answer**")
                    st.write(getattr(ans, "response", str(ans)))
                    if getattr(ans, "source_nodes", None):
                        with st.expander("Sources"):
                            for i, sn in enumerate(ans.source_nodes, 1):
                                st.markdown(f"**{i}.** {sn.node.get_content(metadata_mode='none')[:500]}…")

                except Exception as e:
                    st.exception(e)
    with colB:
        if st.button("(Re)Ingest sample docs"):
            try:
                from rag_setup import ingest_texts
                ingest_texts([
                    "Friendli serves open-source models behind an OpenAI-compatible API.",
                    "Weaviate stores vectors so agents can retrieve relevant context.",
                    "LlamaIndex orchestrates retrieval and tool-calling for agents.",
                ])
                st.success("Sample docs ingested into Weaviate.")
            except Exception as e:
                st.exception(e)

st.caption(f"Backend: {FASTAPI_URL}")
