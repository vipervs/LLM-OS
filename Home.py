import nest_asyncio
import streamlit as st

nest_asyncio.apply()

st.set_page_config(
    page_title="Graph Gate",
    page_icon=":orange_heart:",
)
st.title("LLM OS Terminal")
st.markdown("> Built with: [langchain][phidata][llamaindex][ollama][streamlit]")

def main() -> None:
    st.markdown("---")
    st.markdown("### Select an AI App from the sidebar:")
    st.markdown("#### * Similarity Search: Find research articles with AI")
    st.markdown("#### * Adaptive RAG")

    st.sidebar.success("Select App from above")
main()
