from utils.llm import get_pipeline
from utils.document_paraser import (
    DocumentSematicSearch,
    get_files_chunks,
    extract_text_from_form_all_file,
    complete_sentence,
)

from utils.embedding import get_top_k_result
from pathlib import Path
import streamlit as st
from utils.converter import get_pdf_files
from utils.constant import (
    folder_location,
    sematic_search_model_names,
)
import math


@st.cache_resource
def extract_document(doc_list):
    return extract_text_from_form_all_file(folder_location, doc_list)


@st.cache_resource
def setup_model(doc_list, document_search_model_name, chunk_token_size):
    prog_bar = st.progress(0, "setup inprogress")
    text = extract_document(doc_list)
    doc_search_model = DocumentSematicSearch(document_search_model_name)

    document_chunks = get_files_chunks(
        doc_search_model.llm_tokenizer, text, chunk_token_size
    )
    print(
        f"sreach model: {sematic_search_model_name},token_size: {chunk_token_size} document all chunks:{len(document_chunks)}"
    )
    batch_size = 50 
    iteration = math.ceil(len(document_chunks)/batch_size)
    start = 0
    doc_embd = []
    for j in range(0,iteration):
        print(iteration-1, j, start, start+batch_size )
        # if iteration == j:
        # temp = doc_search_model.get_document_embedding(document_chunks[start:])
        # else:
        temp = doc_search_model.get_document_embedding(document_chunks[start: start+batch_size ])
        doc_embd.extend(temp)
        start = start+batch_size

    print("embedding array lenght", len(doc_embd) )
    prog_bar.progress(100, "setup inprogress")
    prog_bar.empty()
    return doc_embd, document_chunks, doc_search_model


if "index" not in st.session_state:
    st.session_state.index = 0

if "token_size" not in st.session_state:
    st.session_state.chunk_token_size = 4

chunk_token_size = st.selectbox(
    "Token Size", range(100, 451, 50), index=st.session_state.chunk_token_size
)

sematic_search_model_name = st.selectbox(
    "Select Semantic Search Model",
    sematic_search_model_names,
    index=st.session_state.index,
)


if sematic_search_model_name != "<select>":
    st.session_state.index = sematic_search_model_names.index(sematic_search_model_name)
    doc_list = tuple(get_pdf_files(folder_location))
    doc_embd, document_chunks, doc_search_model = setup_model(
        doc_list, sematic_search_model_name, chunk_token_size
    )


question = st.text_input(
    "question",
    value="",
    max_chars=None,
    key="llm",
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
if question:
    query_emb = doc_search_model.get_document_embedding(question)
    topk_docs = doc_search_model.get_topk_result(query_emb, doc_embd, 10)
    result = " ".join([document_chunks[doc_info[0]] for doc_info in topk_docs])
    st.write(
        [complete_sentence(document_chunks[doc_info[0]]) for doc_info in topk_docs]
    )
