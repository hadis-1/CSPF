"""Record the exact revision (commit hash) and last-modified date of every evaluated checkpoint.
Usage (needs internet): python record_model_revisions.py ../docs/model_registry.csv ../docs/model_revisions.csv
Run it once and keep model_revisions.csv with the results; to pin later runs, pass revision=<sha> to from_pretrained."""
import sys, pandas as pd
from huggingface_hub import HfApi
reg = pd.read_csv(sys.argv[1]); api = HfApi(); rows = []
for _, r in reg.iterrows():
    try:
        i = api.model_info(r.huggingface_checkpoint); rows.append(dict(model=r.model, checkpoint=r.huggingface_checkpoint, sha=i.sha, last_modified=str(i.last_modified)))
    except Exception as e: rows.append(dict(model=r.model, checkpoint=r.huggingface_checkpoint, sha="ERROR", last_modified=str(e)[:80]))
for extra in ["MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", "sentence-transformers/all-MiniLM-L6-v2", "gpt2", "distilroberta-base", "bert-base-uncased", "roberta-base"]:
    try: i = api.model_info(extra); rows.append(dict(model="(auxiliary)", checkpoint=extra, sha=i.sha, last_modified=str(i.last_modified)))
    except Exception as e: rows.append(dict(model="(auxiliary)", checkpoint=extra, sha="ERROR", last_modified=str(e)[:80]))
pd.DataFrame(rows).to_csv(sys.argv[2], index=False); print(pd.DataFrame(rows).to_string(index=False))
