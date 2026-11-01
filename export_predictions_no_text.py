"""Prepares the per-pair prediction file for publication WITHOUT tweet text.
Usage: python export_predictions_no_text.py ALL_MODELS_combined_FIXED.csv out_dir [--max-mb 90]
Drops every text column (names containing 'review', 'text', 'tweet', 'perturbed'), keeps keys and numeric/label columns,
writes one gzip-compressed CSV per source dataset (files stay below GitHub's 100 MB limit; if a file is larger it is split into parts).
Prints the columns it kept and dropped so that you can check them before uploading."""
import sys, os, pandas as pd
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
df = pd.read_csv(src, low_memory=False)
drop = [c for c in df.columns if any(k in c.lower() for k in ['review', 'text', 'tweet', 'perturbed'])]
keep = [c for c in df.columns if c not in drop]
print('DROPPED (text):', drop); print('KEPT:', keep)
assert any('dataset' in c.lower() for c in keep) or 'source_dataset' in df.columns, 'no dataset column found'
key = 'source_dataset' if 'source_dataset' in df.columns else [c for c in keep if 'dataset' in c.lower()][0]
if key not in keep: keep = [key] + keep
for name, g in df[keep].groupby(key):
    n = max(1, int(g.memory_usage(deep=True).sum() / 1e6 / 250)) ; parts = [g.iloc[i::n] for i in range(n)] if n > 1 else [g]
    for i, p in enumerate(parts):
        fn = os.path.join(out, f'predictions_{name}' + (f'_part{i+1}' if len(parts) > 1 else '') + '.csv.gz'); p.to_csv(fn, index=False, compression='gzip'); print(fn, round(os.path.getsize(fn) / 1e6, 1), 'MB')
