from tika import parser
import pandas as pd
import os
import sys
import re
import json


def remove_urls_within_parentheses(string):
    pattern = r"\(https?://.*?\)"
    cleaned_string = re.sub(pattern, "", string)
    return cleaned_string


def get_pdf_files(dir_name):
    list_pdf = []
    list_file = os.listdir(dir_name)
    for file in list_file:
        if file.endswith("pdf"):
            list_pdf.append(file)
    return list_pdf


def read_json_file(fpath):
    if fpath.exists():
        with open(f"{fpath}", "r") as f:
            return json.load(f)
    return {}


def save_json(fpath, dictionary):
    json_object = json.dumps(dictionary, indent=4)
    # fpath = os.path.join(dir_name, fname)
    with open(f"{fpath}", "w") as outfile:
        outfile.write(json_object)


def delete_documents(dir_name):
    files = get_pdf_files(dir_name)
    for f in files:
        os.remove(os.path.join(dir_name, f))


def pdf_converter(
    directory_path, fname=None, min_length=300, include_line_breaks=False
):
    """
    Function to convert PDFs to Dataframe with columns as title & paragraphs.

    Parameters
    ----------

    min_length : integer
        Minimum character length to be considered as a single paragraph

    include_line_breaks: bool
        To concatenate paragraphs less than min_length to a single paragraph



    Returns
    -------------
    df : Dataframe


    Description
    -----------------
    If include_line_breaks is set to True, paragraphs with character length
    less than min_length (minimum character length of a paragraph) will be
    considered as a line. Lines before or after each paragraph(length greater
    than or equal to min_length) will be concatenated to a single paragraph to
    form the list of paragraphs in Dataframe.

    Else paragraphs are appended directly to form the list.

    """
    list_pdf = []
    if fname:
        list_pdf = [fname]
        pass
    else:
        list_pdf = get_pdf_files(directory_path)
    final_para_list = []

    df = pd.DataFrame(columns=["title", "paragraphs"])
    for i, pdf in enumerate(list_pdf):
        # try:
        df.loc[i] = [pdf.replace(".pdf", ""), None]
        raw = parser.from_file(os.path.join(directory_path, pdf))
        s = raw["content"].strip()
        paragraphs = re.split("\n\n(?=\u2028|[A-Z-0-9])", s)
        list_par = []
        temp_para = ""  # variable that stores paragraphs with length<min_length
        # (considered as a line)
        for p in paragraphs:
            if not p.isspace():  # checking if paragraph is not only spaces
                if include_line_breaks:  # if True, check length of paragraph
                    if len(p) >= min_length:
                        if temp_para:
                            # if True, append temp_para which holds concatenated
                            # lines to form a paragraph before current paragraph p
                            list_par.append(temp_para.strip())
                            temp_para = (
                                ""  # reset temp_para for new lines to be concatenated
                            )
                            list_par.append(
                                p.replace("\n", "")
                            )  # append current paragraph with length>min_length
                        else:
                            list_par.append(p.replace("\n", ""))
                    else:
                        # paragraph p (line) is concatenated to temp_para
                        line = p.replace("\n", " ").strip()
                        temp_para = temp_para + f" {line}"
                else:
                    # appending paragraph p as is to list_par
                    list_par.append(p.replace("\n", ""))
            else:
                if temp_para:
                    list_par.append(temp_para.strip())

        # df.loc[i, "paragraphs"] = list_par
        # list_par = [doc in list_par]
        final_para_list.extend(list_par)
    return final_para_list

    # except:
    #     print("Unexpected error:", sys.exc_info()[0])
    #     print("Unable to process file {}".format(pdf))
    # return df
