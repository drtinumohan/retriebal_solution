#%%
import os
from utils.constant import sematic_search_model_names, model_dir
from transformers import AutoTokenizer, AutoModel
# %%
for model_name in sematic_search_model_names[1:]:
    print(model_name)
    fpath = os.path.basename(os.path.normpath(model_name))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(f"{model_dir}{fpath}")
    tokenizer.save_pretrained(f"{model_dir}{fpath}")
# %%
model_name = AutoModel.from_pretrained(f"{model_dir}{fpath}")
# %%
