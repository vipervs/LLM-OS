from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from typing_extensions import TypedDict
from typing import List
from langchain.schema import Document
from langgraph.graph import END, StateGraph
from langchain_exa import ExaSearchRetriever, TextContentsOptions
from langchain_community.document_loaders import PyPDFLoader
import streamlit as st
import os

llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
web_search_tool = ExaSearchRetriever(k=3, text_contents_options=TextContentsOptions(max_length=200))

# Streamlit
st.title("Adaptive RAG 🧠🔄📚")
# Create two columns with a ratio of 2:1
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

    # Summarize the documents to understand their topics
    document_text = " ".join([doc.page_content for doc in combined_chunks])

    # Truncate the document text to fit within the token limit
    max_tokens = 8192 
    truncated_document_text = document_text[:max_tokens]

    summary_prompt = PromptTemplate(
        template="Summarize the following text to understand its main topics:\n\n{text}",
        input_variables=["text"],
    )
    summary = summary_prompt.invoke({"text": truncated_document_text})

    # Add to vectorDB
    vectorstore = Chroma.from_documents(
        documents=combined_chunks,
        collection_name="rag-chroma", 
        embedding=embeddings,
    )

    question = user_input
    retriever = vectorstore.as_retriever()
    
    routing_prompt = PromptTemplate(
        template="""You are an expert at routing a user question to a vectorstore or web search. \n
        Here is the summary of the loaded documents: {summary}\n
        Based on this summary, use the vectorstore for questions relevant to the topics mentioned in the summary. \n
        Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question. \n
        Return the JSON with a single key 'datasource' and no preamble or explanation. \n
        Question to route: {question}""",
        input_variables=["summary", "question"],
    )

    question_router = routing_prompt | llm | JsonOutputParser()
    docs = retriever.get_relevant_documents(question)
    doc_txt = docs[1].page_content
    st.write("_Route question to:_")
    st.write(question_router.invoke({"summary": summary, "question": question}))

    prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
            Here is the retrieved document: \n\n {document} \n\n
            Here is the user question: {question} \n
            If the document contains keywords related to the user question, grade it as relevant. \n
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
            Provide the binary score as a JSON with a single key 'score' and no premable or explaination.""",
            input_variables=["question", "document"],
        )

    retrieval_grader = prompt | llm | JsonOutputParser()
    docs = retriever.get_relevant_documents(question)
    doc_txt = docs[1].page_content
    #st.write("_Question is relevant to document:_")
    #st.write(retrieval_grader.invoke({"question": question, "document": doc_txt}))

    ### Generate
    prompt = hub.pull("rlm/rag-prompt-llama3")

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    generation = rag_chain.invoke({"context": docs, "question": question})

    ### Hallucination Grader 

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
        Here are the facts:
        \n ------- \n
        {documents} 
        \n ------- \n
        Here is the answer: {generation}
        Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
        input_variables=["generation", "documents"],
    )

    hallucination_grader = prompt | llm | JsonOutputParser()
    #st.write("_Answer is grounded:_")
    #st.write(hallucination_grader.invoke({"documents": docs, "generation": generation}))

    ### Answer Grader 

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is useful to resolve a question. \n 
        Here is the answer:
        \n ------- \n
        {generation} 
        \n ------- \n
        Here is the question: {question}
        Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
        input_variables=["generation", "question"],
    )

    answer_grader = prompt | llm | JsonOutputParser()
    #st.write("_Answer is useful to resolve question:_")
    #st.write(answer_grader.invoke({"question": question,"generation": generation}))

    ### Question Re-writer

    # Prompt 
    re_write_prompt = PromptTemplate(
        template="""You a question re-writer that converts an input question to a better version that is optimized
        for vectorstore retrieval. Look at the initial and formulate an improved question.
        Here is the initial question: \n\n {question}. \n
        Improved question with no preamble or explanation: \n""",
        input_variables=["question"],
    )

    question_rewriter = re_write_prompt | llm | StrOutputParser()
    question_rewriter.invoke({"question": question})

    class GraphState(TypedDict):
        """
        Represents the state of our graph.

        Attributes:
            question: Users question
            generation: LLM generation
            documents: List of documents 
        """
        question : str
        generation : str
        documents : List[str]

    def retrieve(state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        documents = retriever.get_relevant_documents(question)
        return {"documents": documents, "question": question}

    def generate(state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        
        # RAG generation
        generation = rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    def grade_documents(state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]
        
        # Score each doc
        filtered_docs = []
        for d in documents:
            score = retrieval_grader.invoke({"question": question, "document": d.page_content})
            grade = score['score']
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return {"documents": filtered_docs, "question": question}

    def transform_query(state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """

        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]

        # Re-write question
        better_question = question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": better_question}

    def web_search(state):
        """
        Web search based on the re-phrased question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with appended web results
        """

        print("---WEB SEARCH---")
        question = state["question"]

        # Web search
        docs = web_search_tool.invoke(question)
        web_results = "\n".join([d.page_content for d in docs])
        web_results = Document(page_content=web_results)

        return {"documents": web_results, "question": question}

    ### Edges ###

    def route_question(state):
        """
        Route question to web search or RAG.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """

        print("---ROUTE QUESTION---")
        question = state["question"]
        print(question)

        source = question_router.invoke({"summary": summary, "question": question})
        print(source)

        if 'datasource' in source:
            datasource = source['datasource']
            print(datasource)

            if datasource == 'web_search':
                print("---ROUTE QUESTION TO WEB SEARCH---")
                return "web_search"
            elif datasource == 'vectorstore':
                print("---ROUTE QUESTION TO RAG---")
                return "vectorstore"
            else:
                print("---UNKNOWN DATASOURCE---")
        else:
            print("---DATASOURCE KEY NOT FOUND---")

        return None

    def decide_to_generate(state):
        print("---ASSESS GRADED DOCUMENTS---")
        filtered_documents = state["documents"]

        if not filtered_documents:
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
            return "transform_query"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"

    def grade_generation_v_documents_and_question(state):
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        score = hallucination_grader.invoke({"documents": documents, "generation": generation})
        grade = score['score']

        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            score = answer_grader.invoke({"question": question,"generation": generation})
            grade = score['score']
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"

    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("web_search", web_search) # web search
    workflow.add_node("retrieve", retrieve) # retrieve
    workflow.add_node("grade_documents", grade_documents) # grade documents
    workflow.add_node("generate", generate) # generatae
    workflow.add_node("transform_query", transform_query) # transform_query

    # Build graph
    workflow.set_conditional_entry_point(
        route_question,
        {
            "web_search": "web_search",
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
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        },
    )

    # Compile
    app = workflow.compile()

    inputs = {"question": user_input}

    for output in app.stream(inputs):
        for key, value in output.items():
            # Node
            st.write(f"Node '{key}':")
            # Optional: print full state at each node
            st.write(value)
        print("\n---\n")

    # Final generation
    st.write(value["generation"])