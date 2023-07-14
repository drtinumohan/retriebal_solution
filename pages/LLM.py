# from utils.llm import get_pipeline
# from langchain import HuggingFacePipeline
# from utils.embedding import  get_top_k_result
# from pathlib import Path
# import torch
# import streamlit as st
# from langchain import PromptTemplate,  LLMChain
# from pages.QA import setup_embed_model
# from utils.converter import pdf_converter, get_pdf_files, read_json_file
# from utils.constant import (
#     folder_location,
#     sematic_search_key_name,
#     qa_key_name,
#     default_sematic_search_model_name,
#     default_qa_model_name,
#     model_config_fname,
#     top_k,
# )

# model_config_fpath = Path(folder_location, model_config_fname)
# model_config = read_json_file(model_config_fpath)
# doc_list = tuple(get_pdf_files(folder_location))
# llm_pipeline = get_pipeline()
# llm = HuggingFacePipeline(pipeline = llm_pipeline)

# emb_model, doc_emb, doc_text, index_model  = setup_embed_model(doc_list, model_config.get(sematic_search_key_name, default_sematic_search_model_name))





# template = """
# read the  below paragraphs and answer the question in a few words
# context: {context} 
# answer of the question {question} 
# """
# prompt = PromptTemplate(template=template, input_variables=["question","context"])

# llm_chain = LLMChain(prompt=prompt, llm=llm)

# # context = """"
# # "State Film Awards (1986)" . Department of Information and Public Relations. Archived from theoriginal on 19 November 2009. Retrieved6 May 2012. Awards See accolades Actor, a Special Jury Mention and a Special Jury Awardfor acting, and an award for Best Feature Film (asproducer), also nine Kerala State Film Awards and 17. Khan, Ujala Ali (14 September 2013). "Reigning southern stars" . The National. Archived from the original on 13 January 2017. Retrieved 11 January 2017. 112. Punnoose, Aby (11 November 2021). "67th-national-film-awards-marakkar-arabikkadalinte-simham-wins-the-best-feature-film-award" . Timesofindia. Retrieved 11 November2021.
# # """ 
# # question = "when mohanlal got kerala state award"
# # prompt.format(question=question,context= context)



# # context = st.text_area(
# #     "context",
# #     value='',
# #     height=300,
# #     max_chars=1500,
# #     key=None,
# #     help=None,
# #     on_change=None,
# #     args=None,
# #     kwargs=None,
# #     placeholder=None,
# #     disabled=False,
# #     label_visibility="visible",
# # )


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
# if question: #and context:
#     query_emb = emb_model.encode(question)
#     top_k_result = get_top_k_result(index_model, query_emb, doc_text, top_k=top_k)
#     relevent_docs = [result[0] for result in top_k_result]
#     context = " ".join(relevent_docs)
   
#     # st.write(llm_chain.run(question=question,context= context))
#     st.write(llm_chain.run(question=question,context= context))
#     st.write(context)
