#%%
import streamlit as st
import os
import pandas as pd
from utils.document_paraser import (
    DocumentSematicSearch
)
from copy import deepcopy
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
def extract_embed_docuemnt(file_path, _doc_search_model):
    data_df = pd.read_excel(file_path)
    answer_headers = list(set(data_df.columns)-set([question_header]))#[val  for val in data_df.columns if val.startswith(answer_header)]
    
    data_df[answer_header] = data_df[answer_headers].apply(lambda row:{col:row[col] for col in answer_headers},axis=1 )
    data_arr = data_df[[question_header]+ [answer_header]].values
    start = 0
    batch_size = 50 
    iteration = math.ceil(len(data_arr)/batch_size)
    doc_embd = []
    for j in range(0,iteration):
        temp = _doc_search_model.get_document_embedding(data_arr[start: start+batch_size, 0].tolist())
        doc_embd.extend(temp)
        start = start+batch_size
    return doc_embd, data_arr, answer_headers
@st.cache_resource
def setup_model(file_paths, sematic_search_model_name):
    meta_data = []
    index = []
    all_doc_embd = []
    all_data_arr = []
    doc_search_model = DocumentSematicSearch(sematic_search_model_name)
    for idx, file_path in enumerate(file_paths):
        abs_path = os.path.join(folder_location, file_path)
        doc_embd, data_arr, answer_headers = extract_embed_docuemnt(abs_path, doc_search_model) 
        all_doc_embd.extend(doc_embd)
        all_data_arr.extend(data_arr)
        meta_data.append(
            {
                "file_name" : file_path,
            }
        )
        index.append(len(data_arr))
    all_data_arr = np.asarray(all_data_arr)
    return all_doc_embd, doc_search_model, all_data_arr, index
       
    



if sematic_search_model_name != '<select>':
    file_paths = get_files(folder_location,"xlsx")
    if not file_paths:
        st.write("No file file found")
    else:
        question_area_disable_flag = False
        doc_embd, doc_search_model, data_arr, index = setup_model(file_paths, sematic_search_model_name)

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
    np_index_arr = np.asarray(index_arr)
    results = np.concatenate((np_index_arr[:,1:],data_arr[np_index_arr[:,0].astype(int),:]), axis = 1)
    top_result = deepcopy(results[0][2])
    top_result.update({"score":results[0][0]})
    st.write(top_result)
    for result in results:
        columns = list(result[2].keys())
        print(columns)
        result[2].update(
            {
                "score":result[0],
                "question":result[1]

            }
        )
        new_cols = ["score","question"]
        df = pd.DataFrame({key: [result[2][key]]for key in result[2]})
        st.dataframe(df[["score","question"]+ list(set(columns)-set(new_cols))])
        # print(result)

    # df = pd.DataFrame(data=temp, columns=["score"]+[question_header]+ [answer_header])
    # st.dataframe(df)
   
    # for ans_col in answer_headers:
    #     df[ans_col] = df[answer_header].apply(lambda row:row[ans_col] )
    # df.drop(columns=answer_header, inplace= True)
    # st.write(df.loc[0][answer_headers].to_dict())
    # st.dataframe(df)
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
