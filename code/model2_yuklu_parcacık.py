    steps: int = 750
    seed: int = 7

    def validate(self):
        assert 1 <= self.total_length_m <= 5
        assert .1 <= self.bore_radius_m <= .75
        assert 0 <= self.solenoid_field_T <= .5
        assert 0 <= self.dipole_field_T <= .5
        assert 0 <= self.kicker_field_T <= .3
        assert 20 <= self.kinetic_energy_MeV <= 500
        assert 0 <= self.initial_half_angle_deg <= 35
        for start, length in ((self.solenoid_start_m,self.solenoid_length_m),
                              (self.dipole_start_m,self.dipole_length_m),
                              (self.kicker_start_m,self.kicker_length_m)):
            assert start >= 0 and length >= 0 and start+length <= self.total_length_m


def momentum_mevc(K, mass):
    return np.sqrt(K*(K+2*mass))


def field_at(z, mode, d):
    """Ideal hard-edge field regions. Axis z; solenoid Bz, dipole/kicker By."""
    B=np.zeros(3)
    if d.solenoid_start_m <= z <= d.solenoid_start_m+d.solenoid_length_m:
        B[2]+=d.solenoid_field_T
    if d.dipole_start_m <= z <= d.dipole_start_m+d.dipole_length_m:
        B[1]+=d.dipole_field_T * (-1 if mode=='B' else 1)
    if mode=='C' and d.kicker_start_m <= z <= d.kicker_start_m+d.kicker_length_m:
        B[1]+=-d.kicker_field_T
    return B


def run(mode='A', d=None, species='muon'):
    """Boris pusher in 3D; fixed energy; tracks start at z=0.
    Returns trajectories and survival weights. No decays are sampled, only weighted.
    Model B = static reversed main dipole; Model C = static opposite kicker state.
    These are steady-state polarity comparisons, NOT transient switching simulations.
    """
    d=d or Design(); d.validate()
    mass,tau=(M_MU_MEV,TAU_MU) if species=='muon' else (M_PI_MEV,TAU_PI)
    rng=np.random.default_rng(d.seed)
    p=momentum_mevc(d.kinetic_energy_MeV,mass)
    gamma=1+d.kinetic_energy_MeV/mass
    beta=np.sqrt(1-1/gamma**2)
    speed=beta*C
    angles=np.deg2rad(d.initial_half_angle_deg)*np.sqrt(rng.uniform(size=d.n_tracks))
    azimuth=rng.uniform(0,2*np.pi,d.n_tracks)
    dirs=np.stack([np.sin(angles)*np.cos(azimuth),np.sin(angles)*np.sin(azimuth),np.cos(angles)],axis=1)
    positions=np.zeros((d.n_tracks,3)); tracks=np.full((d.n_tracks,d.steps+1,3),np.nan)
    tracks[:,0,:]=positions
    # step time chosen so near-axial particles traverse length in steps
    dt=d.total_length_m/(d.steps*speed)
    # relativistic rotation frequency qB/(gamma*m), expressed via p and speed
    omega_per_T=299.792458*speed/p  # radians per second per tesla, p in MeV/c
    alive=np.ones(d.n_tracks,dtype=bool)
    exit_index=np.full(d.n_tracks,d.steps)
    for k in range(d.steps):
        for i in np.where(alive)[0]:
            z=positions[i,2]
            if z>=d.total_length_m or np.hypot(*positions[i,:2])>d.bore_radius_m or z<0:
                alive[i]=False; exit_index[i]=k; continue
            B=field_at(z,mode,d)
            t=0.5*omega_per_T*dt*B
            vminus=dirs[i]
            vprime=vminus+np.cross(vminus,t)
            vplus=vminus+np.cross(vprime,2*t/(1+np.dot(t,t)))
            dirs[i]=vplus/np.linalg.norm(vplus)
            positions[i]+=dirs[i]*speed*dt
            tracks[i,k+1]=positions[i]
    elapsed=exit_index*dt
    survival=np.exp(-elapsed/(gamma*tau))
    exit_z=np.array([tracks[i, min(exit_index[i],d.steps),2] for i in range(d.n_tracks)])
    # End directions are not necessarily aligned with detector; no neutrino emission simulated.
    return dict(tracks=tracks, directions=dirs, survival=survival,
                exit_z=exit_z, time_s=elapsed, initial_count=d.n_tracks,
                lost_fraction=np.mean(exit_index<d.steps),
                survival_mean=np.mean(survival),
                gamma=gamma, decay_length_m=beta*gamma*C*tau)


def plot_comparison(d=None, outfile='geometry_comparison.png'):
    d=d or Design()
    fig,axs=plt.subplots(1,3,figsize=(15,4.3),sharex=True,sharey=True)
    for ax,mode in zip(axs,'ABC'):
        result=run(mode,d)
        for track in result['tracks'][:40]:
            ax.plot(track[:,2],track[:,0],lw=.7,alpha=.55)
        ax.axhline(d.bore_radius_m,color='gray',ls='--',lw=1)
        ax.axhline(-d.bore_radius_m,color='gray',ls='--',lw=1)
        ax.axvspan(d.solenoid_start_m,d.solenoid_start_m+d.solenoid_length_m,alpha=.08)
        ax.axvspan(d.dipole_start_m,d.dipole_start_m+d.dipole_length_m,alpha=.12)
        if mode=='C': ax.axvspan(d.kicker_start_m,d.kicker_start_m+d.kicker_length_m,alpha=.16)
        ax.set_title(f'Model {mode} | bore exits {result["lost_fraction"]:.1%}')
        ax.set_xlabel('z (m)')
        ax.grid(alpha=.2)
    axs[0].set_ylabel('x (m)')
    fig.suptitle('Illustrative charged-muon trajectories; NOT neutrino beams')
    fig.tight_layout()
    fig.savefig(outfile,dpi=160)
    plt.close(fig)

if __name__=='__main__':
    d=Design()
    out=Path(__file__).parent
    plot_comparison(d,str(out/'geometry_comparison.png'))
    print('Illustrative static-polarity tests (not ON/OFF performance):')
    for species in ('pion','muon'):
        for mode in 'ABC':
            r=run(mode,d,species)
            print(f'{species:5s} {mode}: decay length={r["decay_length_m"]:.2f} m; '
                  f'bore-exit fraction={r["lost_fraction"]:.3f}; '
                  f'mean survival={r["survival_mean"]:.4f}')
