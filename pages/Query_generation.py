from utils.llm import get_pipeline
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
    qa_key_name,
    default_sematic_search_model_name,
    default_qa_model_name,
    model_config_fname,
    top_k,
)
llm_pipeline = get_pipeline()
llm = HuggingFacePipeline(pipeline = llm_pipeline)


prompt_area = st.text_area(
    "Prompt templete",
    value='',
    height=200,
    max_chars=350,
    key=None,
    help=None,
    on_change=None,
    args=None,
    kwargs=None,
    placeholder=None,
    disabled=False,
    label_visibility="visible",
)

input_area = st.text_input(
    "input_variable",
    value='["context","question"]',
    max_chars=50,
    key=None,
    help=None,
    on_change=None,
    args=None,
    kwargs=None,
    placeholder=None,
    disabled=False,
    label_visibility="visible",
)


query_area = st.text_area(
    "query",
    value = '{"context":"""""","question":""""""}',
    height=200,
    max_chars=15000,
    key=None,
    help=None,
    on_change=None,
    args=None,
    kwargs=None,
    placeholder=None,
    disabled=False,
    label_visibility="visible",
)

query_area = eval(query_area)

if prompt_area and input_area and query_area.get("context") and query_area.get("question"):
    # print(eval(prompt_area))
    # print(eval(prompt_area)[-5:])
    template = eval(prompt_area)
    input_variables = eval(input_area)
    print(template,input_variables )
    prompt = PromptTemplate(template=template, input_variables=input_variables)

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    # print(eval(query_area))
    st.write(llm_chain.run(**query_area))
