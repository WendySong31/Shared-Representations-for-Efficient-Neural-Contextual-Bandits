"""Reproducible stationary/drifting stream experiments and saved per-round rewards."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
from bandits import neural_run, baseline_run

def run_one(job):
    args, algorithm, condition, seed = job
    data = np.load(args['data']); x, y = data['x'], data['y']
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y)).astype(np.int64)
    start = int(rng.integers(len(y)))
    drift = args['drift_every'] if condition == 'drift' else 0
    out = Path(args['output']) / condition / algorithm
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f'seed_{seed}.npz'
    config = {**args, 'algorithm': algorithm, 'condition': condition, 'seed': seed,
              'start': start, 'numpy': np.__version__, 'python': platform.python_version()}
    if dest.exists():
        existing = np.load(dest)
        old = json.loads(str(existing['metadata']))
        if old == config:
            return f'Skipped matching completed run {dest}'
        raise ValueError(f'{dest} belongs to different settings; choose a new output directory.')
    begin = time.perf_counter()
    if algorithm.startswith('nb'):
        variant = int(algorithm[-1])
        pairs = [(args['nb1_hidden'],args['nb1_rate'])] if variant == 1 else [
            (h,r) for h in (1,5,25,50,100) for r in (.01,.1,1.)]
        sizes = np.array([h for h,r in pairs],dtype=np.int64)
        rates = np.array([r for h,r in pairs],dtype=np.float64)
        rewards, logw, selections = neural_run(x,y,order,start,args['rounds'],sizes,rates,
            variant,seed,args['gamma'],args['model_gamma'],drift,
            args['nb3_feedback']=='all', args['nb2_propensity']=='marginal')
    else:
        rewards = baseline_run(x,y,order,start,args['rounds'],
            {'banditron':0,'linucb':1,'cts':2}[algorithm],seed,args['gamma'],drift,
            args['alpha'],args['ts_scale'])
        logw = np.empty((0,)); selections = np.empty((0,))
    seconds = time.perf_counter()-begin
    # Atomic write: interrupted experiments do not masquerade as finished results.
    temporary = dest.with_suffix('.tmp.npz')
    np.savez_compressed(temporary,rewards=rewards,logw=logw,selections=selections,
                        metadata=json.dumps(config,sort_keys=True),seconds=seconds)
    temporary.replace(dest)
    acc = rewards[-min(100000,len(rewards)):].mean()
    return f'{condition}/{algorithm}/seed={seed}: final-window accuracy={acc:.4f}; {seconds:.1f}s'

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data',default='data/covertype.npz')
    p.add_argument('--output',default='results/paper')
    p.add_argument('--rounds',type=int,default=2000000)
    p.add_argument('--seeds',type=int,nargs='+',default=list(range(10)))
    p.add_argument('--algorithms',nargs='+',choices=['nb1','nb2','nb3','banditron','linucb','cts'],
                   default=['nb1','nb2','nb3','banditron','linucb','cts'])
    p.add_argument('--conditions',nargs='+',choices=['stationary','drift'],default=['stationary','drift'])
    p.add_argument('--gamma',type=float,default=.005)
    p.add_argument('--model-gamma',type=float,default=.1)
    p.add_argument('--drift-every',type=int,default=500000)
    p.add_argument('--nb1-hidden',type=int,default=25)
    p.add_argument('--nb1-rate',type=float,default=.1)
    p.add_argument('--nb3-feedback',choices=['all','selected'],default='all')
    p.add_argument('--nb2-propensity',choices=['conditional','marginal'],default='conditional')
    p.add_argument('--alpha',type=float,default=1.)
    p.add_argument('--ts-scale',type=float,default=.1)
    p.add_argument('--workers',type=int,default=1)
    a = p.parse_args()
    if a.rounds<=0 or a.drift_every<=0 or a.workers<=0 or a.nb1_hidden<=0:
        p.error('rounds, drift-every, workers, and nb1-hidden must be positive')
    if not 0<a.gamma<=.5 or not 0<a.model_gamma<=.5:
        p.error('exploration parameters must be in (0, .5]')
    config = vars(a).copy()
    config['data_sha256'] = hashlib.sha256(Path(a.data).read_bytes()).hexdigest()
    jobs = [(config,alg,cond,seed) for cond in a.conditions for alg in a.algorithms for seed in a.seeds]
    print(f'{len(jobs)} runs × {a.rounds:,} sequential rounds; outputs: {a.output}',flush=True)
    # Compilation happens in each process on first call; use modest worker counts.
    if a.workers == 1:
        for job in jobs:
            print(run_one(job),flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures = [pool.submit(run_one,j) for j in jobs]
            for future in concurrent.futures.as_completed(futures):
                print(future.result(),flush=True)

if __name__ == '__main__':
    main()
