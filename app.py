# app.py
import streamlit as st
from agent import agent

st.set_page_config(page_title="Policy Q&A Agent", page_icon="🤖")
st.title("🤖 Policy Q&A Copilot")

if "chat" not in st.session_state:
    st.session_state.chat = []

for role, msg in st.session_state.chat:
    st.chat_message(role).markdown(msg)

if prompt := st.chat_input("Ask a policy question…"):
    st.session_state.chat.append(("user", prompt))
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent.chat(prompt)
            st.markdown(response)
    st.session_state.chat.append(("assistant", response))
