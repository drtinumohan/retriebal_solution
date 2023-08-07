folder_location = "download_location/"
model_config_fname = "model_config.json"
model_dir="model_files/"
default_sematic_search_model_name = "sentence-transformers/msmarco-MiniLM-L-6-v3"
default_qa_model_name = "deepset/roberta-base-squad2-distilled"
top_k = 5

sematic_search_model_names = (
    '<select>',
    default_sematic_search_model_name,
    "sentence-transformers/msmarco-MiniLM-L-12-v3",
    "sentence-transformers/msmarco-distilbert-base-v3",
    "sentence-transformers/msmarco-distilbert-base-v4",
    'intfloat/e5-base-v2',
    'sentence-transformers/all-mpnet-base-v2'
)

qa_model_names = (
    default_qa_model_name,
    "deepset/roberta-base-squad2",
    "deepset/deberta-v3-large-squad2",
    "tiiuae/falcon-7b-instruct",
)

sematic_search_key_name = "sematic_search_model_name "
qa_key_name = "qa_model_name"
question_header = "question"
answer_header = "Answer"
