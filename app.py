import streamlit as st
from dotenv import load_dotenv
import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Load API key from .env
load_dotenv()
HF_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# --- Streamlit UI setup ---
st.set_page_config(page_title="Simple LangChain Chatbot 🤖", layout="centered")
st.title("💬 Simple LangChain Chatbot (Hugging Face)")
st.caption("Just for fun — powered by Hugging Face Endpoint & LangChain")

# --- Load Model (cached) ---
@st.cache_resource
def load_model():
    llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-20b",  # You can change model here
        task="text-generation",
        huggingfacehub_api_token=HF_API_KEY
    )
    return ChatHuggingFace(llm=llm)

model = load_model()

# --- Chat Input ---
user_input = st.chat_input("Type your message...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    result = model.invoke(user_input)

    with st.chat_message("assistant"):
        st.markdown(result.content)
