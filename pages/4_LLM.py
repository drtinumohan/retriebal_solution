from utils.llm import get_pipeline
from utils.document_paraser import (
    DocumentSematicSearch,
    get_files_chunks,
    extract_text_from_form_all_file,
    get_text_chunks,
    complete_sentence,
)
from langchain import HuggingFacePipeline
from utils.embedding import get_top_k_result
from pathlib import Path
import torch
import streamlit as st
from langchain import PromptTemplate, LLMChain

from utils.converter import pdf_converter, get_pdf_files, read_json_file
from utils.constant import (
    sematic_search_model_names,
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


@st.cache_resource
def setup_model(doc_list, doc_search_model_name, chunk_token_size):
    text = extract_text_from_form_all_file(folder_location, doc_list)
    doc_search_model = DocumentSematicSearch(doc_search_model_name)
    document_chunks = get_files_chunks(
        doc_search_model.llm_tokenizer, text, chunk_token_size
    )
    doc_embd = doc_search_model.get_document_embedding(document_chunks)
    return doc_embd, document_chunks, doc_search_model


doc_list = tuple(get_pdf_files(folder_location))

if "index" not in st.session_state:
    st.session_state.index = 3

if "token_size" not in st.session_state:
    st.session_state.chunk_token_size = 3

chunk_token_size = st.selectbox(
    "Token Size", range(100, 301, 50), index=st.session_state.chunk_token_size
)

sematic_search_model_name = st.selectbox(
    "Select Semantic Search Model",
    sematic_search_model_names,
    index=st.session_state.index,
)

doc_embd, document_chunks, doc_search_model = setup_model(
    doc_list,
    sematic_search_model_name,
    chunk_token_size,
)

llm_pipeline = get_pipeline()
llm = HuggingFacePipeline(pipeline=llm_pipeline)
# template = """Answer the question as truthfully as possible using the provided text, and if the answer is not contained within the text below, respond with "I can't answer that"
# Context:
# {context}
# QUESTION:  {question}
# """
# template = """You are a helpful AI assistant and provide the answer for the question based on the given context.
# Context:{context}

# >>QUESTION<<{question}
# >>ANSWER<<"""
template = """You are a helpful AI assistant and provide answers as truthfully as by possible using the given context while disregarding the case sensitivity of the characters in the context.
Context:{context}

>>QUESTION<<{question}
>>ANSWER<<"""
prompt = PromptTemplate(template=template, input_variables=["question", "context"])
llm_chain = LLMChain(prompt=prompt, llm=llm)


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
if question:  # and context:
    query_emb = doc_search_model.get_document_embedding(question)
    topk_docs = doc_search_model.get_topk_result(query_emb, doc_embd, k=2)

    result = [complete_sentence(document_chunks[doc_info[0]]) for doc_info in topk_docs]
    context = " ".join(result)
    print("LLM \n", prompt.format(question=question, context=context))
    st.write(llm_chain.run(question=question, context=context))
    # st.write(result)

    # query_emb = emb_model.encode(question)
    # top_k_result = get_top_k_result(index_model, query_emb, doc_text, top_k=top_k)
    # relevent_docs = [result[0] for result in top_k_result]
    # context = " ".join(relevent_docs)

    # st.write(llm_chain.run(question=question,context= context))
    # st.write(context)
