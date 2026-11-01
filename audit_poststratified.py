"""Post-stratified audit estimates (Section 4.4, Appendix A.14, A.27) from the released label table.
Usage:  python audit_poststratified.py ../audit/audit_labels_576_no_text.csv ../audit/corpus_counts_family_severity.csv OUTDIR
Writes audit_poststrat.csv (annotator 1, annotator 2, both-yes) and validity_auc.csv (AUC of Phi / M1 / NLI against each label)."""
import sys, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
H = pd.read_csv(sys.argv[1]); N = pd.read_csv(sys.argv[2]); out = sys.argv[3]; os.makedirs(out, exist_ok=True)
st = H.groupby(['perturbation_family','sev']).size().rename('n').reset_index().merge(N, on=['perturbation_family','sev'], how='left'); st['w'] = st.N/st.n
H = H.merge(st[['perturbation_family','sev','w']], on=['perturbation_family','sev']); H['stratum'] = H.perturbation_family+'_'+H.sev.astype(str)
rng = np.random.default_rng(1); groups = [np.where(H.stratum.values == s)[0] for s in H.stratum.unique()]; w = H.w.values
ps = lambda idx, col: (col[idx]*w[idx]).sum()/w[idx].sum(); res = {}
for c in ['y1','y2','yc']:
    col = H[c].values; est = ps(np.arange(len(H)), col); bs = [ps(np.concatenate([g[rng.integers(0,len(g),len(g))] for g in groups]), col) for _ in range(2000)]
    res[c] = (est, *np.percentile(bs,[2.5,97.5]), col.mean()); print(c, np.round(res[c],3))
pd.DataFrame(res, index=['ps','lo','hi','raw']).T.to_csv(f'{out}/audit_poststrat.csv')
rows = []
for sn, s in {'Phi':H.sem_preservation_score.values, 'M1':H.m1_cosine_sim.values, 'nli_bi':H.nli_bi.values}.items():
    for ln, y in {'annotator 1':H.y1.values, 'annotator 2':H.y2.values, 'both yes':H.yc.values}.items():
        a = roc_auc_score(y, s); bs = []
        for _ in range(1000):
            i = rng.integers(0,len(y),len(y))
            if y[i].min() != y[i].max(): bs.append(roc_auc_score(y[i], s[i]))
        rows.append((sn, ln, round(a,3), *np.round(np.percentile(bs,[2.5,97.5]),3)))
A = pd.DataFrame(rows, columns=['score','label','AUC','lo','hi']); print(A.to_string()); A.to_csv(f'{out}/validity_auc.csv', index=False)
