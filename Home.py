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


if "sematic_search_model_name" not in st.session_state:
    st.session_state.sematic_search_model_name = 0

if "qa_model_name" not in st.session_state:
    st.session_state.qa_model_name = 0


sematic_search_model_name = st.selectbox(
    "Select Semantic Search Model", 
    sematic_search_model_names,
    index= st.session_state.sematic_search_model_name
)
qa_model_name = st.selectbox("QA Model", 
                                qa_model_names,
                            index= st.session_state.qa_model_name
                                )
config_Submit = st.button(label="Submit")

if config_Submit and sematic_search_model_name !=  "<select>" and qa_model_name!=   "<select>":
    st.session_state.sematic_search_model_name = sematic_search_model_names.index(sematic_search_model_name)
    save_path = Path(folder_location, model_config_fname)
    model_config = {
        sematic_search_key_name: sematic_search_model_name,
        qa_key_name: qa_model_name,
    }
    st.success(
        f"model {sematic_search_model_name} and {qa_model_name} is successfully saved!"
    )
    save_json(save_path, model_config)
elif config_Submit:
    st.write("select above configrations")
else:
    pass

