from utils.llm import get_pipeline
from utils.document_paraser import DocumentSematicSearch, get_files_chunks, extract_text_from_form_all_file, get_text_chunks
from langchain import HuggingFacePipeline
from utils.embedding import  get_top_k_result
from pathlib import Path
import torch
import streamlit as st
from langchain import PromptTemplate,  LLMChain
from utils.converter import pdf_converter, get_pdf_files, read_json_file
from utils.constant import (
    folder_location,
    sematic_search_key_name,
    sematic_search_model_names,
    qa_key_name,
    default_sematic_search_model_name,
    default_qa_model_name,
    model_config_fname,
    top_k,
)

@st.cache_resource
def extract_document(doc_list):
    return extract_text_from_form_all_file(folder_location, doc_list)

@st.cache_resource
def setup_model(doc_list, document_search_model_name, chunk_token_size):
    prog_bar = st.progress(0, "setup inprogress")
    text = extract_document(doc_list)#extract_text_from_form_all_file(folder_location, doc_list)
    doc_search_model = DocumentSematicSearch(document_search_model_name)
    document_chunks = get_files_chunks(doc_search_model.tokenizer, text, chunk_token_size)
    doc_embd = doc_search_model.get_document_embedding(document_chunks)
    prog_bar.progress(100, "setup inprogress")
    prog_bar.empty()
    return doc_embd, document_chunks, doc_search_model



if "index" not in st.session_state:
    st.session_state.index = 0

if "token_size" not in st.session_state:
    st.session_state.chunk_token_size = 0

chunk_token_size = st.selectbox(
    "Token Size", 
    range(100,301,50),
    index = st.session_state.chunk_token_size 
)

sematic_search_model_name = st.selectbox(
    "Select Semantic Search Model", 
    sematic_search_model_names,
    index = st.session_state.index 
)



if sematic_search_model_name !="<select>":
    print(sematic_search_model_name, chunk_token_size )
    st.session_state.index  = sematic_search_model_names.index(sematic_search_model_name)
    doc_list = tuple(get_pdf_files(folder_location))
    doc_embd, document_chunks, doc_search_model = setup_model(doc_list, sematic_search_model_name, chunk_token_size)



question = st.text_input(
    "question",
    value='',
    max_chars=None,
    key='llm',
    type="default",
    help=None,
    autocomplete=None,
    on_change=None,
    args=None,
    kwargs=None,
    placeholder=None,
    disabled=False,
    label_visibility="visible",
)
if question: #and context:

    query_emb  = doc_search_model.get_document_embedding(question)
    topk_docs = doc_search_model.get_topk_result(query_emb, doc_embd)
    result = " ".join([document_chunks[doc_info[0]] for doc_info in topk_docs])
    st.write([document_chunks[doc_info[0]] for doc_info in topk_docs])
