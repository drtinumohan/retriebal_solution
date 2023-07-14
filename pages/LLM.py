from utils.llm import get_pipeline
from utils.document_paraser import DocumentSematicSearch, get_files_chunks, extract_text_from_form_all_file, get_text_chunks
from langchain import HuggingFacePipeline
from utils.embedding import  get_top_k_result
from pathlib import Path
import torch
import streamlit as st
from langchain import PromptTemplate,  LLMChain
from pages.QA import setup_embed_model
from utils.converter import pdf_converter, get_pdf_files, read_json_file
from utils.constant import (
    folder_location,
    sematic_search_key_name,
    qa_key_name,
    default_sematic_search_model_name,
    default_qa_model_name,
    model_config_fname,
    top_k,
)

model_config_fpath = Path(folder_location, model_config_fname)
model_config = read_json_file(model_config_fpath)
model_config.get(sematic_search_key_name, default_sematic_search_model_name)

doc_list = tuple(get_pdf_files(folder_location))

@st.cache_resource
def setup_model(doc_list):
    text = extract_text_from_form_all_file(folder_location, doc_list)
    doc_search_model = DocumentSematicSearch(model_config.get(sematic_search_key_name, default_sematic_search_model_name))
    document_chunks = get_files_chunks(doc_search_model.tokenizer, text)
    doc_embd = doc_search_model.get_document_embedding(document_chunks)
    return doc_embd, document_chunks, doc_search_model


llm_pipeline = get_pipeline()
llm = HuggingFacePipeline(pipeline = llm_pipeline)






template = """
read the  below paragraphs and answer the question in a few words
context: {context} 
answer of the question {question} 
"""
prompt = PromptTemplate(template=template, input_variables=["question","context"])

llm_chain = LLMChain(prompt=prompt, llm=llm)
doc_embd, document_chunks, doc_search_model = setup_model(doc_list)
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


    # query_emb = emb_model.encode(question)
    # top_k_result = get_top_k_result(index_model, query_emb, doc_text, top_k=top_k)
    # relevent_docs = [result[0] for result in top_k_result]
    # context = " ".join(relevent_docs)
   
    # # st.write(llm_chain.run(question=question,context= context))
    # st.write(llm_chain.run(question=question,context= context))
    # st.write(context)
