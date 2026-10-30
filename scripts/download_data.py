from huggingface_hub import hf_hub_download

from paiec.harness.data import AUX, FOLDS, RAW, REV, TABLES

for b in FOLDS + AUX:
    for t in TABLES:
        hf_hub_download("aims-foundations/measurement-db", f"{b}/formatted_tables/{t}.parquet", repo_type="dataset",
                        revision=REV, local_dir=RAW)
print(RAW)
