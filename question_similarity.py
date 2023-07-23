#%%
import streamlit as st
import os
import pandas as pd
from utils.document_paraser import (
    DocumentSematicSearch
)
from utils.converter import get_files
from utils.constant import (
    question_header,
    answer_header,
    sematic_search_model_names,
    folder_location,
    sematic_search_key_name,
    qa_key_name,
    default_sematic_search_model_name,
    default_qa_model_name,
    model_config_fname,
    top_k,
)
import math
import numpy as np
#%%
question_area_disable_flag = True
sematic_search_model_name = st.selectbox(
    "Select Semantic Search Model",
    sematic_search_model_names,
    index=0,
)
@st.cache_resource
def setup_model(file_paths, sematic_search_model_name):
    file_path = os.path.join(folder_location, file_paths[0])
    data_df = pd.read_excel(file_path)
    answer_headers = [val  for val in data_df.columns if val.startswith(answer_header)]
    doc_search_model = DocumentSematicSearch(sematic_search_model_name)
    data_arr = data_df[[question_header]+ answer_headers].values
    start = 0
    batch_size = 50 
    iteration = math.ceil(len(data_arr)/batch_size)
    doc_embd = []
    for j in range(0,iteration):
        temp = doc_search_model.get_document_embedding(data_arr[start: start+batch_size, 0].tolist())
        doc_embd.extend(temp)
        start = start+batch_size
    return doc_embd, doc_search_model, data_arr, answer_headers


if sematic_search_model_name != '<select>':
    file_paths = get_files(folder_location,"xlsx")
    if not file_paths:
        st.write("No file file found")
    else:
        question_area_disable_flag = False
        doc_embd, doc_search_model, data_arr, answer_headers = setup_model(file_paths, sematic_search_model_name)

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
    disabled=question_area_disable_flag,
    label_visibility="visible",
)
if question:
    query = doc_search_model.get_document_embedding([question])
    index_arr = doc_search_model.get_topk_result(query, doc_embd,k=3)
    st.write(data_arr[index_arr[0][0],-1])
    np_index_arr = np.asarray(index_arr)
    temp = np.concatenate((np_index_arr[:,1:],data_arr[np_index_arr[:,0].astype(int),:]), axis = 1)
    # print(data_arr[np_index_arr[:,0].astype(int),:].shape, np_index_arr[:,1].shape)
    # st.write(data_arr[np_index_arr[:,0].astype(int),:])
    df = pd.DataFrame(data=temp, columns=["score"]+[question_header]+ answer_headers)
    st.dataframe(df)
    # st.write(temp)

# %%
# file_path = '/home/boss/working_env/nlp_preprocessing/dataset/AC_Nov_2022.xlsx'
# df = pd.read_excel(file_path,header= None, names= ["Req. ID","Requirement","Vendor Response","Vendor Comments"])
# qtn_col_name =["Requirement"]
# response_cols = ["Vendor Response","Vendor Comments"]
# #%%
# data_arr = df[qtn_col_name+response_cols].values#.tolist()
# # %%
# doc_search_model = DocumentSematicSearch("sentence-transformers/all-mpnet-base-v2")

# #%%
# batch_size = 50 
# iteration = math.ceil(len(data_arr)/batch_size)
# start = 0
# doc_embd = []
# for j in range(0,iteration):
#     print(iteration-1, j, start, start+batch_size )
#     # if iteration == j:
#     # temp = doc_search_model.get_document_embedding(document_chunks[start:])
#     # else:
#     temp = doc_search_model.get_document_embedding(data_arr[start: start+batch_size, 0].tolist())
#     doc_embd.extend(temp)
#     start = start+batch_size
# # %%
# def get_result(query):
#     query = doc_search_model.get_document_embedding([query])
#     index_arr = doc_search_model.get_topk_result(query, doc_embd,k=1)
#     return data_arr[index_arr[0][0],-1]
# # print()
# # %%
# get_result("how your api are protected?")
# # %%
