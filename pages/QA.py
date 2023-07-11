from utils.converter import pdf_converter,get_pdf_files
from utils.embedding import get_model,get_embedding,get_top_k_result
import streamlit as st
from utils.constant import folder_location
from utils.qa_model import get_qa_model, get_anwser, format_qa_output



@st.cache_resource
def setup_embd_model(doc_list):
    doc_text = pdf_converter(folder_location)
    emb_model = get_model()
    doc_emb = get_embedding(emb_model,doc_text)
    return emb_model, doc_emb, doc_text

@st.cache_resource
def setup_qa_model():
    qa_model = get_qa_model()
    return qa_model


doc_list = tuple(get_pdf_files(folder_location))
emb_model, doc_emb, doc_text  =  setup_embd_model(doc_list)
qa_model = setup_qa_model()
text_input = st.text_input(
    "question",
    value='',
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

# if st.button("execute"):
if text_input:
    query_emb = emb_model.encode(text_input)
    top_k_result = get_top_k_result(doc_emb, query_emb, doc_text)
    relevent_docs =[result[0] for result in top_k_result]
    document_relevance_scores = [result[1] for result in top_k_result]
    result_dict = get_anwser(qa_model, text_input, relevent_docs )
    result_df =format_qa_output(relevent_docs, document_relevance_scores, result_dict)
    st.dataframe(result_df[["answer","paragraph","score","re_score","start","end"]], width=1000) 

