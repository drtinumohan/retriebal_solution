import streamlit as st
from pathlib import Path
from utils.converter import save_json

from utils.constant import (
    folder_location,
    sematic_search_model_names,
    qa_model_names,
    sematic_search_key_name,
    qa_key_name,
    model_config_fname,
)

st.title("Model selection")

with st.form(
    key="model_selection", clear_on_submit=True
):  # st.form(key="Form :", clear_on_submit = True):
    sematic_search_model_name = st.selectbox(
        "Select Semantic Search Model", sematic_search_model_names
    )
    qa_model_name = st.selectbox("QA Model", qa_model_names)
    config_Submit = st.form_submit_button(label="Submit")

if config_Submit:
    save_path = Path(folder_location, model_config_fname)
    model_config = {
        sematic_search_key_name: sematic_search_model_name,
        qa_key_name: qa_model_name,
    }
    st.success(
        f"model {sematic_search_model_name} and {qa_model_name} is successfully saved!"
    )
    save_json(save_path, model_config)
    # st.write(model_config)
