import streamlit as st
import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

try:
    HF_API_KEY = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
except Exception:
    load_dotenv()
    HF_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HF_API_KEY:
    st.error("❌ Hugging Face API key not found! Please set it in Streamlit Secrets (for Cloud) or in .env (for local).")
    st.stop()

st.set_page_config(page_title="Simple LangChain Chatbot 🤖", layout="centered")
st.title("💬 Simple LangChain Chatbot (Hugging Face)")
st.caption("Just for fun — powered by Hugging Face Endpoint & LangChain")

@st.cache_resource
def load_model():
    llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-20b",
        task="text-generation",
        huggingfacehub_api_token=HF_API_KEY
    )
    return ChatHuggingFace(llm=llm)

model = load_model()

user_input = st.chat_input("Type your message...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("🤔 Thinking..."):
        try:
            result = model.invoke(user_input)
            response = result.content
        except Exception as e:
            response = f"⚠️ Error: {e}"

    with st.chat_message("assistant"):
        st.markdown(response)
