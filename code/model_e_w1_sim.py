"""Model E W1 A1/A2/A3: fully synthetic counts, not detector predictions."""
import numpy as np
from dataclasses import dataclass

FS=10; WINDOW=600; BIT_SAMPLES=4; PACKET_BITS=14; PACKET_SAMPLES=56
SYNC=[1,0,1,0]
RATES={'solar':0.025,'cosmic':0.018,'detector':0.18} # counts per sample, synthetic
SIGNAL=1.3 # additional counts per active OOK sample, synthetic
P_BURST=0.002; BURST_MEAN=5.0
N_WINDOWS=400; SEED=20261109

def encode(data):
    return np.array(SYNC+list(data)+list(map(int,f'{sum(data)%4:02b}')),dtype=int)

def decode(bits):
    bits=np.asarray(bits,dtype=int)
    return np.array_equal(bits[:4],SYNC) and (int(''.join(map(str,bits[12:14])),2)==sum(bits[4:12])%4)

def poisson_bit_llr(samples,b,s):
    # log p(n|1)/p(n|0) with known background mean and signal mean per sample
    counts=samples.sum(); n=len(samples)
    return counts*np.log((b+s)/b)-n*s

def main():
    rng=np.random.default_rng(SEED)
    n=N_WINDOWS*WINDOW
    solar=rng.poisson(RATES['solar'],n)
    cosmic=rng.poisson(RATES['cosmic'],n)
    detector=rng.poisson(RATES['detector'],n)
    bursts=(rng.random(n)<P_BURST)*rng.poisson(BURST_MEAN,n)
    env=1.0
    background=solar+detector+rng.poisson(env*RATES['cosmic'],n)+bursts
    # 1 packet per window, random start aligned with 4 samples; independent false-sync scan in idle windows
    tx=np.zeros(n,dtype=int); active=np.zeros(n,dtype=bool); slots=[]
    for w in range(N_WINDOWS):
        start=w*WINDOW+int(rng.integers(15,WINDOW//BIT_SAMPLES-15-PACKET_BITS))*BIT_SAMPLES
        data=rng.integers(0,2,8); bits=encode(data)
        for j,bit in enumerate(bits):
            if bit: active[start+j*BIT_SAMPLES:start+(j+1)*BIT_SAMPLES]=True
        slots.append((start,bits,data))
    signal=rng.poisson(SIGNAL*active)
    observed=background+signal
    # Calibration using separate independent preamble-free interval (synthetic; no oracle leakage)
    calibration=rng.poisson(sum(RATES.values()),6000)
    initial=float(calibration.mean())
    # Build a causal per-window background estimate. Use previous window's A1/A2/A3 for decoding current window.
    estimates={'A1':initial,'A2':initial,'A3':initial}
    stats={k:dict(bit_err=0,bit_total=0,packet_err=0,packet_total=0,false_accept=0,miss=0,abs_error=[],squared_error=[],means=[]) for k in estimates}
    # Candidate search: true slot and 4 decoy idle starts per window; test sync + checksum and evidence threshold.
    # No oracle packet flags in decoder. For unbiased comparison, every method sees same candidates.
    for w in range(N_WINDOWS):
        left=w*WINDOW; right=left+WINDOW; obs=observed[left:right]
        truth_bg=float(background[left:right].mean())
        true_start,true_bits,data=slots[w]
        true_local=true_start-left
        decoys=[]
        for st in rng.permutation(np.arange(0,WINDOW-PACKET_SAMPLES+1,BIT_SAMPLES)):
            if abs(int(st)-true_local)>PACKET_SAMPLES+BIT_SAMPLES:
                decoys.append(int(st))
                if len(decoys)==4:break
        candidates=[true_local]+decoys
        for key in stats:
            b=max(estimates[key],.02)
            decisions=[]
            for st in candidates:
                chunk=obs[st:st+PACKET_SAMPLES].reshape(PACKET_BITS,BIT_SAMPLES)
                llr=np.array([poisson_bit_llr(row,b,SIGNAL) for row in chunk])
                bits=(llr>0).astype(int)
                valid=decode(bits)
                # require positive correlation to preamble, not just binary hard bits
                evidence=float(sum((2*np.array(SYNC)-1)*llr[:4]))
                accepted=bool(valid and evidence>2.0)
                decisions.append((st,bits,accepted,llr))
            true_dec=decisions[0]
            decoded=true_dec[1]
            valid_true=true_dec[2]
            S=stats[key]
            S['bit_err']+=int(np.count_nonzero(decoded[4:12]!=data)); S['bit_total']+=8
            S['packet_err']+=int(not (valid_true and np.array_equal(decoded[4:12],data)))
            S['packet_total']+=1
            S['miss']+=int(not valid_true)
            S['false_accept']+=sum(int(d[2]) for d in decisions[1:])
            # A2: replace positive-peak samples in ACCEPTED packets by prior baseline.
            # A3: additionally replace positive peaks in candidates with sync-like signal but failed checks by A1.
            adjusted=obs.astype(float).copy()
            raw_mean=float(obs.mean())
            if key!='A1':
                for st,bits,accepted,llr in decisions:
                    if accepted:
                        for j in range(PACKET_BITS):
                            if bits[j]==1:
                                seg=slice(st+j*BIT_SAMPLES,st+(j+1)*BIT_SAMPLES)
                                adjusted[seg]=np.where(adjusted[seg]>b,b,adjusted[seg])
                    elif key=='A3' and np.array_equal(bits[:4],SYNC):
                        for j in range(PACKET_BITS):
                            seg=slice(st+j*BIT_SAMPLES,st+(j+1)*BIT_SAMPLES)
                            adjusted[seg]=np.where(adjusted[seg]>raw_mean,raw_mean,adjusted[seg])
            new_mean=float(adjusted.mean())
            estimates[key]=new_mean
            S['means'].append(new_mean)
            S['abs_error'].append(abs(new_mean-truth_bg))
            S['squared_error'].append((new_mean-truth_bg)**2)
    print('SIMULATION PARAMETERS')
    print(f'windows={N_WINDOWS}, seconds={N_WINDOWS*60}, samples={n}, packet_bits=14, bit_samples={BIT_SAMPLES}')
    print(f'background rates/sample={RATES}, burst probability/sample={P_BURST}, burst mean={BURST_MEAN}, signal/sample on={SIGNAL}')
    print(f'Observed true background mean={background.mean():.5f}; observed signal mean={signal.mean():.5f}; calibration={initial:.5f}')
    print('METHOD BER PER FALSE_ACCEPT/DECOY MISS/TRUE RMSE_BACKGROUND MAE_BACKGROUND MEAN_EST')
    for k,S in stats.items():
        print(k, f"{S['bit_err']/S['bit_total']:.5f}",f"{S['packet_err']/S['packet_total']:.5f}",f"{S['false_accept']/(4*N_WINDOWS):.5f}",f"{S['miss']/N_WINDOWS:.5f}",f"{np.sqrt(np.mean(S['squared_error'])):.5f}",f"{np.mean(S['abs_error']):.5f}",f"{np.mean(S['means']):.5f}")
if __name__=='__main__': main()
