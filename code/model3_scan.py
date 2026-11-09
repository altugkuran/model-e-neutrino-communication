"""Illustrative magnetic-layout scan; not a neutrino source or validated yield model.
Requires numpy, matplotlib. Run: python model3_scan.py
Uses relativistic Boris rotation, uniform monoenergetic positive pions, ideal hard-edge fields.
Scores only charged-pion survival and exit-direction acceptance; NOT neutrino counts.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
C=299792458.0; MPI=139.57039; TAUPI=2.6033e-8
N=1800; STEPS=450; LENGTH=5.; RADIUS=.55; KINETIC=100.; CONE_DEG=12.
SEED=42

def sample():
    rng=np.random.default_rng(SEED)
    # cone centered 7 degrees toward +x, to make a dipole steering problem nontrivial
    a=np.deg2rad(20)*np.sqrt(rng.random(N)); phi=2*np.pi*rng.random(N)
    v=np.stack([np.sin(a)*np.cos(phi),np.sin(a)*np.sin(phi),np.cos(a)],axis=1)
    tilt=np.deg2rad(7)
    v[:,[0,2]]=np.stack([v[:,0]*np.cos(tilt)+v[:,2]*np.sin(tilt),-v[:,0]*np.sin(tilt)+v[:,2]*np.cos(tilt)],axis=1)
    return v

INITIAL=sample()

def propagate(solstart,sollen,dipstart,diplen,kickstart,kicklen,mode,state,Bsol=.15,Bdip=.015,Bkick=.020):
    p=np.sqrt(KINETIC*(KINETIC+2*MPI)); gamma=1+KINETIC/MPI
    beta=np.sqrt(1-1/gamma**2); speed=beta*C
    dt=LENGTH/(240*speed); omega=299.792458*speed/p
    pos=np.zeros((N,3)); vel=INITIAL.copy(); active=np.ones(N,bool)
    t_alive=np.zeros(N); dead=np.zeros(N,bool)
    for _ in range(STEPS):
        ix=np.where(active)[0]
        if len(ix)==0: break
        z=pos[ix,2]; B=np.zeros((len(ix),3))
        B[:,2]=Bsol*((z>=solstart)&(z<solstart+sollen))
        dip_sign=state if mode=='B' else 1
        B[:,1]=Bdip*dip_sign*((z>=dipstart)&(z<dipstart+diplen))
        if mode=='C': B[:,1]+=Bkick*state*((z>=kickstart)&(z<kickstart+kicklen))
        t=.5*omega*dt*B; vm=vel[ix]; vp=vm+np.cross(vm,t)
        vnew=vm+np.cross(vp,2*t/(1+np.sum(t*t,axis=1))[:,None])
        vel[ix]=vnew/np.linalg.norm(vnew,axis=1)[:,None]
        pos[ix]+=vel[ix]*speed*dt; t_alive[ix]+=dt
        escaped=(np.linalg.norm(pos[ix,:2],axis=1)>RADIUS)|(pos[ix,2]<0)
        exited=pos[ix,2]>=LENGTH
        active[ix[escaped|exited]]=False
        dead[ix[escaped]]=True
    survived=np.exp(-t_alive/(gamma*TAUPI))
    exitmask=(pos[:,2]>=LENGTH)&(~dead)
    cosang=vel[:,2] # receiver centered along +z; proxy for charged particles only
    accepted=exitmask & (cosang>=np.cos(np.deg2rad(CONE_DEG)))
    return float(np.mean(survived*accepted)),float(np.mean(exitmask)),float(np.mean(dead))

def main():
    rows=[]
    # 3 solenoid placements x 3 dipole placements x 3 kicker placements; valid nonoverlap
    for solstart in (.0,.3,.6):
      sollen=1.2
      for dipstart in (1.5,2.,2.5):
       diplen=.65
       for kickstart in (3.2,3.7,4.2):
        kicklen=.45
        if dipstart+diplen>kickstart: continue
        for mode in 'ABC':
          plus,ep,lp=propagate(solstart,sollen,dipstart,diplen,kickstart,kicklen,mode,1)
          minus,em,lm=propagate(solstart,sollen,dipstart,diplen,kickstart,kicklen,mode,-1)
          contrast=(plus-minus)/(plus+minus) if plus+minus>0 else 0
          # score retains on acceptance while rewarding contrast; arbitrary illustrative objective
          score=plus*max(0,contrast)
          rows.append(dict(model=mode,solenoid_start_m=solstart,dipole_start_m=dipstart,kicker_start_m=kickstart,
                           on_acceptance=round(plus,6),alternative_acceptance=round(minus,6),
                           contrast=round(contrast,6),score=round(score,6),on_exit_fraction=round(ep,6),on_wall_loss=round(lp,6)))
    with open(OUT/'scan.csv','w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    best=[]
    for mode in 'ABC':
      choices=[r for r in rows if r['model']==mode]
      selected=max(choices,key=lambda r:r['score']) if mode!='A' else max(choices,key=lambda r:r['on_acceptance'])
      best.append(selected)
      print(mode,selected)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    labels=['A fixed','B reverse','C kicker']
    x=np.arange(3)
    axs[0].bar(x-.17,[r['on_acceptance'] for r in best],.34,label='ON geometry')
    axs[0].bar(x+.17,[r['alternative_acceptance'] for r in best],.34,label='Alternative geometry')
    axs[0].set_xticks(x,labels);axs[0].set_ylabel('Pion exit acceptance (per produced pion)');axs[0].legend(fontsize=8)
    axs[1].bar(x,[r['contrast'] for r in best]);axs[1].set_xticks(x,labels);axs[1].set_ylabel('Geometric contrast (not signal contrast)')
    fig.suptitle('Toy layout scan — not neutrino flux or actual gamma OFF behavior');fig.tight_layout();fig.savefig(OUT/'scan.png',dpi=160)
    with open(OUT/'best.csv','w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(best)

if __name__=='__main__': main()
