from utils.llm import get_pipeline
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
llm_pipeline = get_pipeline()
llm = HuggingFacePipeline(pipeline = llm_pipeline)
template = """
{prompt} 
"""
prompt = PromptTemplate(template=template, input_variables=["prompt"])

llm_chain = LLMChain(prompt=prompt, llm=llm)




context = st.text_area(
    "context",
    value="""
    input:  toyota camary le 
    output: {"make_name":"toyota", "model_name": "camary", "trim_name":"le"}. 
    input: kia seltos rt 
    output: {"make_name":"kia", "model_name": "seltos", "trim_name":"rt"}. 
    predict the output in json format for the input  maruthi swift desire

    """,
    height=300,
    max_chars=1500,
    key=None,
    help=None,
    on_change=None,
    args=None,
    kwargs=None,
    placeholder=None,
    disabled=False,
    label_visibility="visible",
)


# question = st.text_input(
#     "question",
#     value='',
#     max_chars=None,
#     key='llm',
#     type="default",
#     help=None,
#     autocomplete=None,
#     on_change=None,
#     args=None,
#     kwargs=None,
#     placeholder=None,
#     disabled=False,
#     label_visibility="visible",
# )
if context:
    st.write(llm_chain.run(prompt=context))
