

import streamlit as st
import streamlit as st
from pathlib import Path
from utils.constant import folder_location 
st.title("QA")
# st.image(res, width = 800)

form1 = st.empty()
pg_bar_val = st.empty()
sc_bar = st.empty()


with form1.form(key="Form :", clear_on_submit = True):#st.form(key="Form :", clear_on_submit = True):
    File = st.file_uploader(label = "Upload file", type=["pdf"])
    Submit = st.form_submit_button(label='Submit')

def progress_bar_ctl(pbar_obj,percentage, text):
    pbar_obj.progress(percentage, text)

def success_bar_ctl(sbar_obj, text):
    sbar_obj.success(text)

if Submit :
    # Save uploaded file to 'F:/tmp' folder.
   
    progress_text = "Document saving in progress. Please wait."
    my_bar = pg_bar_val.progress(0, text=progress_text)
    save_path = Path(folder_location, File.name)
    with open(save_path, mode='wb') as w:
        w.write(File.getvalue())

    if save_path.exists():
        my_bar = pg_bar_val.progress(100, text="Done")
        success_bar = sc_bar.success(f'File {File.name} is successfully saved!')
    # progress_bar_ctl(my_bar, 100,"Document extration in progress")
    # array_list = pdf_converter(save_folder,fname=File.name, min_length=200)
    # success_bar_ctl(success_bar, f'File {File.name} extration is successfully Done!')

        # form1.empty()
        # my_bar.empty()
        # success_bar.empty()


# text_input = st.text_input(
#     "question",
#     value='',
#     max_chars=None,
#     key=None,
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

# if st.button("execute"):
#     example = spark.createDataFrame([[text_input, text_area]]).toDF(
#         "question", "context"
#     )
#     result = pipeline.fit(example).transform(example)import streamlit as st
# from utils.converter import pdf_converter
# from utils.embedding import get_model,get_embedding
# import streamlit as st
# from pathlib import Path


# st.title("QA")
# # st.image(res, width = 800)

# form1 = st.empty()
# pg_bar_val = st.empty()
# sc_bar = st.empty()

# with form1.form(key="Form :", clear_on_submit = True):#st.form(key="Form :", clear_on_submit = True):
#     File = st.file_uploader(label = "Upload file", type=["pdf"])
#     Submit = st.form_submit_button(label='Submit')

# def progress_bar_ctl(pbar_obj,percentage, text):
#     pbar_obj.progress(percentage, text)

# def success_bar_ctl(sbar_obj, text):
#      sbar_obj.success(text)

# if Submit :
#     # Save uploaded file to 'F:/tmp' folder.
#     save_folder = 'download_location/'
#     progress_text = "Document saving in progress. Please wait."
#     my_bar = pg_bar_val.progress(0, text=progress_text)
#     save_path = Path(save_folder, File.name)
#     with open(save_path, mode='wb') as w:
#         w.write(File.getvalue())

#     if save_path.exists():
#         success_bar = sc_bar.success(f'File {File.name} is successfully saved!')
#     progress_bar_ctl(my_bar, 25,"Document extration in progress")
#     array_list = pdf_converter(save_folder,fname=File.name, min_length=200)
#     success_bar_ctl(success_bar, f'File {File.name} extration is successfully Done!')
#     progress_bar_ctl(my_bar, 50,"Geting embeddings")
#     emd_model = get_model()
#     docment_emb = get_embedding(emd_model, array_list)
#     progress_bar_ctl(my_bar, 100,"Done")
#     success_bar_ctl(success_bar, f'Done')
#     form1.empty()
#     my_bar.empty()
#     success_bar.empty()


# # text_input = st.text_input(
# #     "question",
# #     value='',
# #     max_chars=None,
# #     key=None,
# #     type="default",
# #     help=None,
# #     autocomplete=None,
# #     on_change=None,
# #     args=None,
# #     kwargs=None,
# #     placeholder=None,
# #     disabled=False,
# #     label_visibility="visible",
# # )

# # if st.button("execute"):
# #     # example = spark.createDataFrame([[text_input, text_area]]).toDF(
# #     #     "question", "context"
# #     # )
# #     # result = pipeline.fit(example).transform(example)
# #     # st.write(result.select("answer.result").toPandas())
# #     st.write(f"{question}{text_input}")




# # # uploaded_file = st.file_uploader("Choose a file", "pdf")
# # # with open(uploaded_file, 'wb') as f: 
# # #     f.write(filebytes)
# #     # st.write(result.select("answer.result").toPandas())
# #     st.write(f"{question}{text_input}")




# # # uploaded_file = st.file_uploader("Choose a file", "pdf")
# # # with open(uploaded_file, 'wb') as f: 
# # #     f.write(filebytes)