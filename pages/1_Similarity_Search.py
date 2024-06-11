import os
import arxiv
import pandas as pd
import json
import streamlit as st
from csv import writer
from scipy import spatial
from glob import glob
import requests
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_community.embeddings import OllamaEmbeddings

# Pydantic Schema for structured response
class Keywords(BaseModel):
    keywords: str = Field(description="The generated keywords in boolean format")

st.set_page_config(page_title="PrivyLens Similarity Search 🔍")
st.title("PrivyLens Similarity Search 🔍")

search_engine = st.selectbox("Select Search Engine:", ["arXiv", "CSE"])

# Function to calculate relatedness between two vectors
def relatedness_function(a, b):
    return 1 - spatial.distance.cosine(a, b)

# Function for making an embedding request with error handling
def get_embedding(text):
    print(f"Requesting embedding for: {text}")
    try:
        embeddings = OllamaEmbeddings(model="snowflake-arctic-embed:latest")
        return embeddings.embed_query(text)
    except Exception as e:
        print(f"Error requesting embedding: {e}")
        return None

# Enhanced arXiv search function with error handling
from search import arxiv_search

from cse_search import google_custom_search

# Function to rank titles based on relatedness
def titles_ranked_by_relatedness(query, source):
    query_embedding = get_embedding(query)

    if source == "arXiv":
        df = pd.read_csv(f'arxiv/{query}.csv', header=None)  
        strings_and_relatedness = [
            (row[0], row[1], row[2], row[3], relatedness_function(query_embedding, json.loads(row[4]))) 
            for i, row in df.iterrows()
        ]
        strings_and_relatedness.sort(key=lambda x: x[4], reverse=True)
    elif source == "CSE":
        df = pd.read_csv(f'cse/{query}.csv', header=None)  
        strings_and_relatedness = [ 
            (row[0], row[1], row[2], relatedness_function(query_embedding, json.loads(row[3]))) 
            for i, row in df.iterrows()
        ]
        strings_and_relatedness.sort(key=lambda x: x[3], reverse=True)
    else:
        raise ValueError(f"Invalid source: {source}")  # Handle unknown sources

    return strings_and_relatedness

# Prompt template for keyword generation
prompt = PromptTemplate.from_template(
    """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a research assistant specializing in generating precise and effective search queries for scientific databases like PubMed, CINAHL, or Web of Science. 

    Given a user query, your task is to craft a comprehensive yet concise boolean search string. 

    Prioritize the following:
    **Accuracy & Relevance:** Understand the user's intent and translate it into a search query that retrieves highly relevant results.
    **Specificity:**  Employ search operators to narrow down results and eliminate irrelevant information.
    **Exhaustiveness:**  Consider synonyms and related terms, to ensure all relevant articles are captured.

    Utilize these advanced search techniques: 
    **Boolean Operators (AND, OR, NOT):** Combine keywords effectively to broaden or narrow your search.
    **Phrase Searching ("..."):** Search for exact phrases and multi-word terms.
    **Truncation (*):** Include variations of keywords by truncating the word stem.

    Follow these steps:
    1. **Analyze the query:** Identify the key concepts and the user's search intent.
    2. **Brainstorm keywords:** Generate a list of relevant keywords, including synonyms and related terms.
    3. **Structure the query:** Combine keywords using Boolean operators and consider the order of operations.
    4. **Apply advanced techniques:** Utilize phrase searching, truncation, proximity operators, and field tags to refine your search.
    5. **Format for readability:** Present the final search string in a clear and easy-to-understand format.

    <|eot_id|><|start_header_id|>user<|end_header_id|>
    QUERY: {query}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
)

# Chain
llm = OllamaFunctions(model="llama3", 
                      format="json", 
                      temperature=0.6)

structured_llm = llm.with_structured_output(Keywords)
chain = prompt | structured_llm

with st.form('search_form'):
    query = st.text_area('Enter text:', max_chars=500)
    if st.form_submit_button('Search'):
        response = chain.invoke({"query": query})
        keywords = response.keywords
        print(f"Generated Keywords: {keywords}")
        st.header(f"📚 Search Results: {keywords}")
        with st.spinner(f"Searching {search_engine}..."):
            if search_engine == "arXiv":
                results = arxiv_search(keywords, get_embedding)
            elif search_engine == "CSE":
                results = google_custom_search(keywords, get_embedding)
            else:
                st.error(f"Unknown search engine: {search_engine}")
                results = []
        
        if not results:
            st.warning(f"No search results found on {search_engine}.")
        else:
            for i, result in enumerate(results, start=1):
                if search_engine == "arXiv":
                    title, summary, published, url, score = result['title'], result['summary'], result['published'], result['pdf_url'], result['relatedness_score']
                    st.subheader(f"Result {i}: {title}")
                    st.write(f"Summary: {summary}")
                    st.write(f"Published: {published}")
                    st.write(f"URL: {url}")
                elif search_engine == "CSE":  
                    title, snippet, url, score = result['title'], result['snippet'], result['link'], result['relatedness_score']
                    st.subheader(f"Result {i}: {title}")
                    st.write(f"Snippet: {snippet}")
                    st.write(f"URL: {url}")
                st.write(f"Relatedness Score: {score:.2f}")
                st.write("---")

# Sidebar sections
st.sidebar.header("Past Searches 📚")
past_searches = glob('arxiv/*.csv') + glob('cse/*.csv')
past_searches_with_folder = [(os.path.dirname(file), os.path.basename(file)) for file in past_searches]
past_search_options = [(folder, file) for folder, file in past_searches_with_folder]

# Group searches by source
searches_by_source = {}
for file_path in past_searches:
    folder = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    source = folder.split('/')[-1]  # Extract source from folder name
    if source not in searches_by_source:
        searches_by_source[source] = []
    searches_by_source[source].append((folder, file_name))

for source, searches in searches_by_source.items():
    with st.sidebar.expander(source):
        for folder, file_name in searches:
            search_label = f"{folder}/{file_name}"
            col1, col2 = st.columns([8, 2])
            with col1:
                if st.checkbox(search_label, key=search_label):
                    if folder == 'arxiv':
                        query = file_name.replace('.csv', '')
                        # Trigger a reload by updating a Streamlit session state variable
                        if 'load_arxiv_results' not in st.session_state:
                            st.session_state['load_arxiv_results'] = query
                    elif folder == 'cse':
                        query = file_name.replace('.csv', '')
                        # Trigger a reload for CSE
                        if 'load_cse_results' not in st.session_state:
                            st.session_state['load_cse_results'] = query
            with col2:
                if st.button("🗑️", key=f"delete_{search_label}"):
                    file_path = os.path.join(folder, file_name)
                    try:
                        os.remove(file_path)
                        st.success(f"Deleted: {search_label}")
                        st.rerun()  # Rerun the app to update the sidebar
                    except FileNotFoundError:
                        st.warning(f"File not found: {search_label}")

# Main window logic to display results based on session state
if 'load_arxiv_results' in st.session_state:
    query = st.session_state['load_arxiv_results']
    file_path = os.path.join('arxiv', query + '.csv') 
    try:
        df = pd.read_csv(file_path, header=None)
        df = df.sort_values(by=4, ascending=False)  
        st.header(f"📚 ArXiv Results: {query}")
        for i, row in df.iterrows():
            title, summary, published, url, score = row[0], row[1], row[2], row[3], row[4]
            st.subheader(f"Result {i + 1}: {title}")
            st.write(f"Summary: {summary}")
            st.write(f"Published: {published}")
            st.write(f"URL: {url}")
            st.write(f"Relatedness Score: {score:.2f}")
            st.write("---")
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        st.warning(f"Error loading past search: {e}")
    finally:
        del st.session_state['load_arxiv_results']  # Clear the state after results are displayed

if 'load_cse_results' in st.session_state:
    query = st.session_state['load_cse_results']
    file_path = os.path.join('cse', query + '.csv') 
    try:
        df = pd.read_csv(file_path, header=None)
        df = df.sort_values(by=3, ascending=False) 
        st.header(f"📚 CSE Results: {query}")
        for i, row in df.iterrows():
            title, snippet, link, score = row[0], row[1], row[2], row[3]
            st.subheader(f"Result {i + 1}: {title}")
            st.write(f"Snippet: {snippet}")
            st.write(f"URL: {link}")
            st.write(f"Relatedness Score: {score:.2f}") 
            st.write("---")
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        st.warning(f"Error loading past search: {e}")
    finally:
        del st.session_state['load_cse_results'] 
