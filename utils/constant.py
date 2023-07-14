folder_location = "download_location/"
model_config_fname = "model_config.json"
default_sematic_search_model_name = "sentence-transformers/all-MiniLM-L6-v2"
default_qa_model_name = "deepset/roberta-base-squad2-distilled"
top_k = 5

sematic_search_model_names = (
    default_sematic_search_model_name,
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-distilroberta-v1",
    "sentence-transformers/msmarco-distilbert-base-tas-b",
)

qa_model_names = (
    default_qa_model_name,
    "deepset/roberta-base-squad2",
    "deepset/deberta-v3-large-squad2",
    "tiiuae/falcon-7b-instruct"
)

sematic_search_key_name = "sematic_search_model_name "
qa_key_name = "qa_model_name"
