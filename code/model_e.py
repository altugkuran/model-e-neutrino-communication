"""Model E: idealized muon extraction scan; no beam optics or photonuclear yield."""
import math,csv
from pathlib import Path
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
C=299792458.; MU=105.6583755; TAU=2.1969811e-6; B=.2; K=100.; T_WAIT=2e-6
GAMMA=1+K/MU; BETA=math.sqrt(1-1/GAMMA**2)
P=math.sqrt(K*(K+2*MU))/1000 # GeV/c
R=P/(.299792458*B); T_REV=2*math.pi*R/(BETA*C); TAU_LAB=GAMMA*TAU
# Per-turn capture p. Extract on each pass, survival between passages.
def evaluate(n,p):
    survive_wait=math.exp(-T_WAIT/TAU_LAB)
    remaining=survive_wait; extracted=0.; decayed=1-survive_wait
    for i in range(n):
        remaining*=math.exp(-T_REV/TAU_LAB)
        extracted+=remaining*p
        remaining*=1-p
    decayed=1-extracted-remaining
    return extracted,remaining,decayed
rows=[]
for p in [.25,.5,.75]:
 for n in range(1,11):
    e,r,d=evaluate(n,p)
    rows.append(dict(capture_probability=p,turns=n,window_ns=n*T_REV*1e9,extracted_initial_pct=e*100,remaining_initial_pct=r*100,decayed_initial_pct=d*100,incremental_vs_previous_pct=(e-evaluate(n-1,p)[0])*100 if n>1 else e*100))
with (ROOT/'results.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
fig,ax=plt.subplots(figsize=(8,4.6))
for p in [.25,.5,.75]:
 subset=[x for x in rows if x['capture_probability']==p]
 ax.plot([x['turns'] for x in subset],[x['extracted_initial_pct'] for x in subset],marker='o',label=f'p={p:.0%}')
ax.set(xlabel='Extraction window (revolutions)',ylabel='Extracted / initial muons (%)',title='Model E: idealized extraction after 2 microseconds')
ax.legend();ax.grid(alpha=.3);fig.tight_layout();fig.savefig(ROOT/'extraction.png',dpi=170)
print(f'gamma={GAMMA:.5f}, beta={BETA:.5f}, radius={R:.4f} m, revolution={T_REV*1e9:.2f} ns, wait survival={math.exp(-T_WAIT/TAU_LAB):.4%}')
for p in [.25,.5,.75]:
 for n in [2,3,4,5]:
  e,r,d=evaluate(n,p);print(f'p={p:.0%} n={n} time={n*T_REV*1e9:.1f} ns extracted={e:.3%} remaining={r:.3%} decayed={d:.3%}')
