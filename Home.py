import nest_asyncio
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["EXA_API_KEY"] = os.getenv("EXA_API_KEY")

nest_asyncio.apply()

st.set_page_config(
    page_title="LLM OS Terminal",
    page_icon=":orange_heart:",
)
st.title("LLM OS Terminal 💻")
st.markdown("> Built with: [langchain][phidata][llamaindex][ollama][streamlit]")

def main() -> None:
    st.markdown("---")
    st.markdown("### Select an AI App from the sidebar:")
    st.markdown("#### * Similarity Search: Find research articles with AI")
    st.markdown("#### * Adaptive RAG")

    st.sidebar.success("Select App from above")
main()
