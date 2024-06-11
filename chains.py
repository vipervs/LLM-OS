from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.runnables import RunnableSequence
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

# Initialize OllamaFunctions with your preferred model and format
llm = OllamaFunctions(model="qwen2", format="json", temperature=0)
#llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")

class SummarizeText(BaseModel):
    summary: str = Field(description="Summarize the given text.")

class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to websearch or a vectorstore.",
    )

class GradeDocuments(BaseModel):
    binary_score: bool = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

class GradeHallucinations(BaseModel):
    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

## Summarize
structured_llm_summarizer = llm.with_structured_output(SummarizeText)
system_summary = """You are an expert summarizer. Summarize the following text to understand its main topics."""
summary_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_summary),
        ("human", "Text: \n\n {text}"),
    ]
)
summary_chain = summary_prompt | structured_llm_summarizer

## Route
structured_llm_router = llm.with_structured_output(RouteQuery)
system_route = """You are an expert at routing a user question to a vectorstore or web search. \n
You will be provided a summary of the vectorstore content as support for routing the question. \n
Use the vectorstore for questions relevant to the topics mentioned in the summary. \n
Otherwise, use websearch."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_route),
        ("human", "Summary: \n\n {summary} \n\n Question to route: {question}"),
    ]
)
question_router = route_prompt | structured_llm_router

## Grade
structured_llm_grader = llm.with_structured_output(GradeDocuments)
system_grade = """You are a grader assessing relevance of a retrieved document to a user question. \n 
If the document contains keywords related to the user question, grade it as relevant. \n
It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_grade),
        ("human", "Retrieved document: \n\n {documents} \n\n User question: {question}"),
    ]
)
retrieval_grader = grade_prompt | structured_llm_grader

## Generate
prompt = hub.pull("rlm/rag-prompt-llama3")
generation_chain = prompt | llm | StrOutputParser()

## Answer
structured_llm_grader = llm.with_structured_output(GradeAnswer)
system_answer = """You are a grader assessing whether an answer addresses / resolves a question \n 
Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_answer),
        ("human", "Answer: {generation} \n\n User question: {question}"),
    ]
)
answer_grader: RunnableSequence = answer_prompt | structured_llm_grader

## Hallucination
structured_llm_grader = llm.with_structured_output(GradeHallucinations)
system_hallucination = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_hallucination),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)
hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader

## Rewriter
system_rewrite = """You a question re-writer that converts an input question to a better version that is optimized for vectorstore retrieval. \n
Look at the initial question and formulate an improved question."""
rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_rewrite),
        ("human", "Here is the initial question: \n\n {question}. Improved question with no preamble or explanation.: \n"),
    ]
)

question_rewriter = rewrite_prompt | llm | StrOutputParser()