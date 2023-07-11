import streamlit as st
import streamlit as st
from pathlib import Path
from utils.constant import (
    folder_location,
    sematic_search_model_names,
    qa_model_names,
    sematic_search_key_name,
    qa_key_name,
    model_config_fname
)
from utils.converter import save_json, delete_documents

st.title("QA")
# st.image(res, width = 800)

form1 = st.empty()
pg_bar_val = st.empty()
sc_bar = st.empty()


def progress_bar_ctl(pbar_obj, percentage, text):
    pbar_obj.progress(percentage, text)


def success_bar_ctl(sbar_obj, text):
    sbar_obj.success(text)


with st.form(
    key="Form :", clear_on_submit=True
):  # st.form(key="Form :", clear_on_submit = True):
    File = st.file_uploader(label="Upload file", type=["pdf"])
    prev_document_flag = st.checkbox('Keep previous documents', value= True)
    file_submit = st.form_submit_button(label="Submit")

with st.form(
    key="model_selection", clear_on_submit=True
):  # st.form(key="Form :", clear_on_submit = True):
    sematic_search_model_name = st.selectbox(
        "Select Semantic Search Model", sematic_search_model_names
    )
    qa_model_name = st.selectbox("QA Model", qa_model_names)
    config_Submit = st.form_submit_button(label="Submit")

    # File = st.file_uploader(label = "Upload file", type=["pdf"])
    # Submit = st.form_submit_button(label='Submit')


if file_submit:
    if prev_document_flag == False: 
        __ = delete_documents(folder_location)

    progress_text = "Document saving in progress. Please wait."
    my_bar = pg_bar_val.progress(0, text=progress_text)
    save_path = Path(folder_location, File.name)
    with open(save_path, mode="wb") as w:
        w.write(File.getvalue())

    if save_path.exists():
        my_bar = pg_bar_val.progress(100, text="Done")
        success_bar = sc_bar.success(f"File {File.name} is successfully saved!")

if config_Submit:
    save_path = Path(folder_location, model_config_fname)
    model_config = {
        sematic_search_key_name: sematic_search_model_name,
         qa_key_name: qa_model_name

    }
    st.success(f"model {sematic_search_model_name} and {qa_model_name} is successfully saved!")
    save_json(save_path,model_config)
    # st.write(model_config)
