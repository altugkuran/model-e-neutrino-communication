"""Model E W1: finite-volume cylinder, vertical muon ring, idealized one-channel interaction.
No neutrino cross-section model or Cherenkov detector response is asserted.
"""
import numpy as np, csv
from scipy.special import roots_legendre
from scipy.constants import pi

M_MU=105.6583755 # MeV/c2
T_MU=100.0
GAMMA=1+T_MU/M_MU
BETA=np.sqrt(1-1/GAMMA**2)
P_MU=np.sqrt((M_MU+T_MU)**2-M_MU**2)/1000 # GeV/c
RING_RADIUS=P_MU/(0.299792458*0.2)
VOLUME=100.0 # m3, 100t water approx
N_FREE_PROTONS=2*(100000/0.01801528)*6.02214076e23
SIGMA_REF_CM2=1e-40
N_SIGMA=N_FREE_PROTONS*SIGMA_REF_CM2
F_DECAY_2US=1-np.exp(-2e-6/(GAMMA*2.1969811e-6))
# quadrature volume: radial 9, azimuth 24, vertical 9
u,wu=roots_legendre(9);u=(u+1)/2;wu=wu/2
z,wz=roots_legendre(9);z=(z+1)/2;wz=wz/2
az=np.arange(24)*2*pi/24

def volume_points(height):
    rad=np.sqrt(VOLUME/(pi*height))
    pts=[];weights=[]
    for ui,wi in zip(u,wu):
      for a in az:
       for zi,wzi in zip(z,wz):
        pts.append([rad*np.sqrt(ui)*np.cos(a),rad*np.sqrt(ui)*np.sin(a),height*(zi-.5)])
        weights.append(wi*wzi/24)
    return np.array(pts),np.array(weights)

def prob_per_decay(height,depth,arc_deg=0):
    pts,w=volume_points(height)
    # Source ring center is at (0,0,-depth); vertical ring lies x-z, tangential upward at phi=0
    if arc_deg==0:
        phi=np.array([0.]);wp=np.array([1.])
    elif arc_deg==360:
        phi=np.linspace(-pi,pi,720,endpoint=False);wp=np.ones(len(phi))/len(phi)
    else:
        phi=np.linspace(-np.deg2rad(arc_deg)/2,np.deg2rad(arc_deg)/2,101);wp=np.ones(len(phi))/len(phi)
    result=0.
    for ph,ww in zip(phi,wp):
        source=np.array([RING_RADIUS*(1-np.cos(ph)),0.,-depth+RING_RADIUS*np.sin(ph)])
        tangent=np.array([np.sin(ph),0.,np.cos(ph)])
        delta=pts-source
        dist=np.linalg.norm(delta,axis=1)
        costheta=(delta@tangent)/dist
        # density per unit area at point: dP/dOmega / dist²
        density=1/(4*pi*GAMMA**2*(1-BETA*costheta)**2)/dist**2
        result+=ww*np.dot(w,density)
    # n*sigma*V = N_targets*sigma, one neutrino flavor per decay
    return result*N_SIGMA*1e-4  # cm² to m²

def main():
    heights=np.arange(.5,10.001,.5)
    depths=[100,250,500,1000,2000,3000,5000]
    arcs=[0,10,30,60,90,360]
    path='/mnt/data/model_e_w1_geometry_results.csv'
    rows=[]
    for depth in depths:
      for h in heights:
        vals={a:prob_per_decay(h,depth,a) for a in arcs}
        rows.append(dict(depth_m=depth,height_m=round(h,2),diameter_m=round(2*np.sqrt(VOLUME/(pi*h)),5),area_m2=round(VOLUME/h,5),**{'p_straight':vals[0]},**{f'p_ring_arc_{a}deg':vals[a] for a in arcs[1:]},p_ring_10deg_all_decays=vals[10]*10/360,p_ring_30deg_all_decays=vals[30]*30/360,p_ring_60deg_all_decays=vals[60]*60/360))
    with open(path,'w',newline='') as f:
      writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('gamma beta ring radius 2us decay fraction',GAMMA,BETA,RING_RADIUS,F_DECAY_2US)
    for d in [100,1000,5000]:
      for h in [0.5,1,2,5,10]:
       r=next(r for r in rows if r['depth_m']==d and r['height_m']==h)
       print(d,h,'straight',f"{r['p_straight']:.4e}",'ring30',f"{r['p_ring_arc_30deg']:.4e}",'full ring',f"{r['p_ring_arc_360deg']:.4e}")
    for d in depths:
      subset=[r for r in rows if r['depth_m']==d]
      best=max(subset,key=lambda r:r['p_straight'])
      spread=max(r['p_straight'] for r in subset)/min(r['p_straight'] for r in subset)
      print('OPT',d,best['height_m'],spread)
if __name__=='__main__':main()
