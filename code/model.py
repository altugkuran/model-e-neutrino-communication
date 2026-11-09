"""Neutrino transmitter conceptual pulse-response model (not a validated beam simulation).
Requires numpy, matplotlib. Run: python model.py
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
OUT=Path(__file__).resolve().parent
# Units: microseconds, arbitrary particle production units per microsecond.
TAU_PI=0.026033; TAU_MU=2.196981
DT=0.002; T_END=240.; PERIOD=40.; DUTY=0.5
Q_ON=1.0
# Illustrative acceptance factors, not measured or derived from a magnet geometry.
A_ON=0.30; A_OFF=0.30
B_ON=0.30; B_OFF=0.045; B_SWITCH_US=10.0
C_ON=0.30; C_OFF=0.06; C_SWITCH_US=0.5
# Fractions of pion and muon neutrinos in an illustrative angular acceptance.
# Can be replaced by angle/energy-dependent functions after particle tracking.
MATERIALS={'carbon':1.0,'beryllium':0.8,'tungsten':1.3}

def simulate(material='carbon', period=PERIOD, duty=DUTY, dt=DT, t_end=T_END):
    n=int(round(t_end/dt)); t=np.arange(n)*dt
    on=(t%period)<period*duty
    q=Q_ON*MATERIALS[material]
    # Exact exponential loss for each time step, stable for short pion lifetime.
    fp=-np.expm1(-dt/TAU_PI); fm=-np.expm1(-dt/TAU_MU)
    npion=0.; nmu=0.
    # Control lag for magnetic acceptance (a phenomenological approximation).
    b=B_OFF; c=C_OFF
    rates={k:np.empty(n) for k in ('A','B','C')}
    pop_pi=np.empty(n); pop_mu=np.empty(n)
    for i in range(n):
        born=q*dt if on[i] else 0.
        # Uniform production within a time step approximated as added at start.
        npion+=born
        d_pi=npion*fp; npion-=d_pi
        nmu+=d_pi
        d_mu=nmu*fm; nmu-=d_mu
        # One pion neutrino and two muon neutrinos per chain in ideal limit.
        r=(d_pi+2*d_mu)/dt
        target_b=B_ON if on[i] else B_OFF
        target_c=C_ON if on[i] else C_OFF
        b+=(target_b-b)*(-np.expm1(-dt/B_SWITCH_US))
        c+=(target_c-c)*(-np.expm1(-dt/C_SWITCH_US))
        rates['A'][i]=r*A_ON
        rates['B'][i]=r*b
        rates['C'][i]=r*c
        pop_pi[i]=npion; pop_mu[i]=nmu
    return t,on,rates,pop_pi,pop_mu

def summarize(t,on,rates,period=PERIOD):
    # Ignore first two cycles to reduce startup transient.
    mask=t>=2*period
    rows=[]
    for name,r in rates.items():
        on_mean=r[mask & on].mean(); off_mean=r[mask & ~on].mean()
        contrast=(on_mean-off_mean)/(on_mean+off_mean) if on_mean+off_mean else 0
        rows.append((name,on_mean,off_mean,contrast))
    return rows

def main():
    t,on,rates,pi,mu=simulate()
    rows=summarize(t,on,rates)
    print('MODEL: arbitrary angular acceptance, not measured magnet performance')
    print('system,ON_mean,OFF_mean,contrast')
    for r in rows: print(f'{r[0]},{r[1]:.6f},{r[2]:.6f},{r[3]:.4f}')
    with (OUT/'summary.csv').open('w',encoding='utf-8') as f:
        f.write('system,ON_mean,OFF_mean,contrast\n')
        for r in rows:f.write(f'{r[0]},{r[1]:.8f},{r[2]:.8f},{r[3]:.6f}\n')
    fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
    axes[0].plot(t,on.astype(float),label='Gamma ON/OFF',color='black');axes[0].set_ylabel('Command');axes[0].set_ylim(-.1,1.1)
    axes[1].plot(t,pi,label='Pion population');axes[1].plot(t,mu,label='Muon population');axes[1].legend();axes[1].set_ylabel('Particles (arb.)')
    for name,r in rates.items():axes[2].plot(t,r,label=name)
    axes[2].legend();axes[2].set_ylabel('Accepted neutrinos (arb./us)');axes[2].set_xlabel('Time (microseconds)')
    for ax in axes:ax.grid(alpha=.2);ax.set_xlim(0,160)
    fig.suptitle('Conceptual pulsed pion-muon transmitter: A / B / C')
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=160)
    plt.close(fig)
if __name__=='__main__':main()
