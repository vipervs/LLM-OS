import streamlit as st
import os
from typing import List, TypedDict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from langgraph.graph import END, StateGraph
from chains import answer_grader, hallucination_grader, question_router, generation_chain, retrieval_grader, question_rewriter, summary_chain

embedding = OllamaEmbeddings(model="snowflake-arctic-embed:latest")
web_search_tool = TavilySearchResults(k=3)

class GraphState(TypedDict):
    question : str
    generation : str
    summary : str
    documents : List[str]

st.title("Adaptive RAG 🧠🔄📚")
col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_input("Question:", placeholder="What is your question?", key='input')

with col2:
    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your files", type=['pdf'], accept_multiple_files=True)
        web_urls = st.text_area("Enter web URLs (one per line)")
        process = st.button("Process")

if process:
    if not uploaded_files and not web_urls:
        st.warning("Please upload at least one PDF file or enter at least one web URL.")
        st.stop()

    text_chunks = []
    web_splits = []

    if uploaded_files:
        temp_dir = '.temp'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        for uploaded_file in uploaded_files:
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_file_path, "wb") as file:
                file.write(uploaded_file.getbuffer())

            try:
                loader = PyPDFLoader(temp_file_path)
                data = loader.load() 
                st.write(f"Data loaded for {uploaded_file.name}")
            except Exception as e:
                st.error(f"Failed to load {uploaded_file.name}: {str(e)}")

            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=800, chunk_overlap=400
            )
            text_chunks = text_splitter.split_documents(data)

    if web_urls:
        urls = web_urls.split("\n")
        try:
            loader = WebBaseLoader(web_paths=urls)
            web_docs = loader.load()
            st.write(f"Loaded {len(web_docs)} documents from web URLs")
        except Exception as e:
            st.error(f"Failed to load web URLs: {str(e)}")

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=800, chunk_overlap=400
        )
        web_splits = text_splitter.split_documents(web_docs)

    combined_chunks = text_chunks + web_splits
    st.write(f"Total number of chunks: {len(combined_chunks)}")

    if not combined_chunks:
        st.warning("No documents found. Please upload at least one PDF file or enter at least one valid web URL.")
        st.stop()

    vectorstore = Chroma.from_documents(
        documents=combined_chunks,
        collection_name="adaptiv-rag",
        persist_directory=".chroma",
        embedding=embedding,
    )

    inputs = {"question": user_input, "documents": combined_chunks}
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    def route_question(state):
        print("---ROUTE QUESTION---")
        question = state["question"]
        summary = state["summary"]
        source = question_router.invoke({"question": question, "summary": summary})
        if source.datasource == "websearch":
            print("---ROUTE QUESTION TO WEB SEARCH---")
            return "websearch"
        elif source.datasource == "vectorstore":
            print("---ROUTE QUESTION TO RAG---")
            return "vectorstore"

    def decide_to_generate(state):
        print("---ASSESS GRADED DOCUMENTS---")
        filtered_documents = state["documents"]
        if not filtered_documents:
            print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
            return "transform_query"
        else:
            print("---DECISION: GENERATE---")
            return "generate"

    def grade_generation_grounded_in_documents_and_question(state):
        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        score = hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )

        if hallucination_grade := score.binary_score:
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            print("---GRADE GENERATION vs QUESTION---")
            score = answer_grader.invoke({"question": question, "generation": generation})
            if answer_grade := score.binary_score:
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"
    
    def summarize(state):
        print("---SUMMARIZE TEXT---")
        documents = state["documents"]
        if documents:
            document_text = " ".join([doc.page_content for doc in documents])
            max_tokens = 8192  # Adjust this based on your model's token limit
            truncated_document_text = document_text[:max_tokens]
            try:
                summary_chain_result = summary_chain.invoke({"text": truncated_document_text})
                print(f"summary_chain_result: {summary_chain_result}")
                summary = summary_chain_result["tool"]
                return {"summary": summary}
            except KeyError as e:
                print(f"KeyError in summary_chain response: {e}")
                print(f"summary_chain_result: {summary_chain_result}")
                return {"summary": "Error: Unable to generate summary"}
        else:
            raise ValueError("No documents found to summarize.")

    def web_search(state):
        print("---WEB SEARCH---")
        question = state["question"]
        # Clear the current documents since we only want web search results
        state["documents"] = []

        docs = web_search_tool.invoke({"query": question})
        web_results = [Document(page_content=d["content"]) for d in docs]
        return {"documents": web_results, "question": question}

    def retrieve(state):
        print("---RETRIEVE---")
        question = state["question"]

        documents = retriever.get_relevant_documents(question)
        return {"documents": documents, "question": question}

    def grade_documents(state):
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        filtered_docs = []
        for d in documents:
            score = retrieval_grader.invoke(
                {"question": question, "documents": d.page_content}
            )
            grade = score.binary_score
            if grade.lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return {"documents": filtered_docs, "question": question}

    def generate(state):
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        generation = generation_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    def transform_query(state):
        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]

        better_question = question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": better_question}

    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("summarize", summarize)
    workflow.add_node("web_search", web_search)
    workflow.add_node("retrieve", retrieve) 
    workflow.add_node("grade_documents", grade_documents) 
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)

    # Build graph
    workflow.set_entry_point("summarize")
    workflow.add_conditional_edges(
        "summarize",
        route_question,
        {
            "websearch": "web_search",
            "vectorstore": "retrieve",
        },
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_grounded_in_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        },
    )

    # Compile the graph
    app = workflow.compile()

    for output in app.stream(inputs):
        for key, value in output.items():
            # Node
            st.write(f"Node '{key}':")
            # Optional: print full state at each node
            st.write(value)
        print("\n---\n")

    # Generate and display the graph image
    st.write(value["generation"])

    graph_image_path = "graph.png"
    app.get_graph().draw_mermaid_png(output_file_path=graph_image_path)
    st.image(graph_image_path, caption='StateGraph Execution')
