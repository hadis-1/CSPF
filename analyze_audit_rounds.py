"""Reproduces the agreement statistics of the second and third audit rounds and the family table
from the label files without tweet text.
Usage: python analyze_audit_rounds.py ../audit/ [out_dir]
Output: Table A46 (round 2 agreement), Table A49 (round 3 agreement), Table A50 (round 3 majority meaning label by family)."""
import sys, itertools, pandas as pd, numpy as np
from sklearn.metrics import cohen_kappa_score as ck
A = sys.argv[1].rstrip('/') + '/'
r2 = pd.read_csv(A + 'round2_labels_no_text.csv'); r2 = r2[~r2.is_control]
print('Round 2 (two annotators, single-question protocol), %d pairs' % len(r2))
for d in ['meaning', 'sentiment', 'natural']:
    x, y = r2[f'{d}_r2a'], r2[f'{d}_r2b']; print(f'  {d:9s} raw {100*(x==y).mean():.1f}%  kappa {ck(x,y):.2f}')
r3 = pd.read_csv(A + 'round3_labels_no_text.csv'); r3 = r3[~r3.is_control].copy(); T = ['r3a', 'r3b', 'r3c']
def meaning(c, s): return np.where(c == 'no', 'no', np.where(s == 'yes', 'yes', 'partial'))
for t in T: r3[f'meaning_{t}'] = meaning(r3[f'content_{t}'], r3[f'strength_{t}'])
print('Round 3 (three annotators, content/strength protocol), %d pairs' % len(r3))
for d in ['content', 'strength', 'meaning', 'sentiment', 'natural']:
    M = r3[[f'{d}_{t}' for t in T]]; pw = [(M[f'{d}_{a}'] == M[f'{d}_{b}']).mean() for a, b in itertools.combinations(T, 2)]
    kk = [ck(M[f'{d}_{a}'], M[f'{d}_{b}']) for a, b in itertools.combinations(T, 2)]
    cats = sorted(set(M.values.ravel())); cnt = np.array([[(row == k).sum() for k in cats] for row in M.values])
    P = ((cnt**2).sum(1) - 3) / 6; Pe = (((cnt.sum(0) / cnt.sum())**2).sum()); fl = (P.mean() - Pe) / (1 - Pe)
    print(f'  {d:9s} pairwise raw {100*min(pw):.0f}-{100*max(pw):.0f}%  kappa {min(kk):.2f}-{max(kk):.2f}  Fleiss {fl:.2f}')
r3['maj'] = r3[[f'meaning_{t}' for t in T]].mode(axis=1)[0]
tab = r3.groupby('perturbation_family').maj.value_counts(normalize=True).unstack(fill_value=0).round(2) * 100
tab['pairs'] = r3.groupby('perturbation_family').size(); print(tab.to_string()); print('overall fully preserving: %.1f%%' % (100 * (r3.maj == 'yes').mean()))
if len(sys.argv) > 2: tab.to_csv(sys.argv[2].rstrip('/') + '/round3_majority_by_family.csv')
