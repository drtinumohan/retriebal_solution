from utils.converter import pdf_converter, get_files, read_json_file
from utils.embedding import get_model, get_embedding, get_top_k_result, get_fasis_model
import streamlit as st
from pathlib import Path
import numpy as np
from utils.constant import (
    folder_location,
    sematic_search_key_name,
    qa_key_name,
    default_sematic_search_model_name,
    default_qa_model_name,
    model_config_fname,
    top_k,
)
from utils.qa_model import get_qa_model, get_anwser, format_qa_output
import time

model_config_fpath = Path(folder_location, model_config_fname)
model_config = read_json_file(model_config_fpath)


# st.write(model_config)

@st.cache_resource
def setup_embed_model(doc_list, sematic_search_model_name):
    # pg_bar_val = st.empty()
    # pg_bar_val.progress(0, text="Extracting Document")
    doc_text = pdf_converter(folder_location)
    # pg_bar_val.progress(25, text="Creating document embeddings")
    emb_model = get_model(sematic_search_model_name)
    doc_emb = get_embedding(emb_model, doc_text)
    index_model = get_fasis_model(doc_emb)
    return  emb_model, doc_emb, doc_text, index_model
    # pg_bar_val.progress(75, text="Creating Faiss index")
    # pg_bar_val.progress(90, text="Loading QA model")


    return emb_model, doc_emb, doc_text, index_model

@st.cache_resource
def setup_model(doc_list, sematic_search_model_name, qa_model_name):
    pg_bar_val = st.empty()
    pg_bar_val.progress(0, text="Extracting Document")
    doc_text = pdf_converter(folder_location)
    # pg_bar_val.progress(25, text="Creating document embeddings")
    emb_model = get_model(sematic_search_model_name)
    doc_emb = get_embedding(emb_model, doc_text)
    index_model = get_fasis_model(doc_emb)
    # pg_bar_val.progress(75, text="Creating Faiss index")
    # pg_bar_val.progress(90, text="Loading QA model")
    emb_model, doc_emb, doc_text, index_model = setup_embed_model(doc_list, sematic_search_model_name)
    qa_model = get_qa_model(qa_model_name)
    pg_bar_val.progress(100, text="Done")
    pg_bar_val.empty()
    return emb_model, doc_emb, doc_text, index_model, qa_model


# @st.cache_resource
# def setup_qa_model(qa_model_name):
#     qa_model = get_qa_model(qa_model_name)
#     return qa_model


doc_list = tuple(get_files(folder_location,"pdf"))
emb_model, doc_emb, doc_text, index_model, qa_model = setup_model(
    doc_list,
    model_config.get(sematic_search_key_name, default_sematic_search_model_name),
    model_config.get(qa_key_name, default_qa_model_name)
)

text_input = st.text_input(
    "question",
    value="",
    max_chars=None,
    key=None,
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
# st.write( model_config.get(sematic_search_key_name, default_sematic_search_model_name),"  " ,model_config.get(qa_key_name, default_qa_model_name))
# if st.button("execute"):
if text_input:
    query_emb = emb_model.encode(text_input)
    # top_k_result = get_top_k_result(doc_emb, query_emb, doc_text,top_k = top_k)
    top_k_result = get_top_k_result(index_model, query_emb, doc_text, top_k=top_k)
    relevent_docs = [result[0] for result in top_k_result]
    st.write("".join(relevent_docs))
    document_relevance_scores = [result[1] for result in top_k_result]
    result_dict = get_anwser(qa_model, text_input, relevent_docs)
    result_df = format_qa_output(relevent_docs, document_relevance_scores, result_dict)
   
  
    st.dataframe(
        result_df[["answer", "paragraph", "score", "re_score", "start", "end"]],
        width=1000,
    )
