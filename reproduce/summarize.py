"""Generate CSV tables and two-panel figures from real saved rewards."""
import argparse
import csv
import json
import os
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='results/paper')
    p.add_argument('--window',type=int,default=100000)
    p.add_argument('--stride',type=int,default=10000)
    a=p.parse_args()
    root=Path(a.input)
    os.environ.setdefault('MPLCONFIGDIR',str((root/'.matplotlib').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups={}
    for file in sorted(root.glob('*/*/seed_*.npz')):
        data=np.load(file)
        meta=json.loads(str(data['metadata']))
        key=(meta['condition'],meta['algorithm'])
        groups.setdefault(key,[]).append((data['rewards'],float(data['seconds']),meta))
    if not groups:
        raise SystemExit('No completed runs found.')
    rows=[]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
    fig2,axes2=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
    for (condition,alg),runs in groups.items():
        n=len(runs); lengths={len(r[0]) for r in runs}
        if len(lengths)!=1:
            raise ValueError('Cannot combine runs with different horizons')
        rewards=np.stack([r[0] for r in runs]).astype(np.float64)
        horizon=rewards.shape[1]
        tail=rewards[:,-min(a.window,horizon):].mean(axis=1)
        se=float(tail.std(ddof=1)/np.sqrt(n)) if n>1 else None
        rows.append({'condition':condition,'algorithm':alg,'runs':n,'rounds':horizon,
            'mean_accuracy':rewards.mean(),'final_window':min(a.window,horizon),
            'final_window_accuracy':tail.mean(),'final_window_seed_sd':tail.std(ddof=1) if n>1 else '',
            'final_window_seed_se':se if se is not None else '',
            'mean_seconds':np.mean([r[1] for r in runs]),
            'paper_reference_regret':.93*horizon-rewards.sum(axis=1).mean(),
            'oracle_mistakes':horizon-rewards.sum(axis=1).mean()})
        steps=np.unique(np.r_[np.arange(a.stride,horizon+1,a.stride),horizon]).astype(int)
        csum=np.cumsum(rewards,axis=1)
        regret=.93*steps[None,:]-csum[:,steps-1]
        panel=0 if condition=='stationary' else 1
        axes[panel].plot(steps,regret.mean(axis=0),label=f'{alg} (n={n})')
        if n>1:
            err=1.96*regret.std(axis=0,ddof=1)/np.sqrt(n)
            axes[panel].fill_between(steps,regret.mean(axis=0)-err,regret.mean(axis=0)+err,alpha=.12)
        lower=np.maximum(0,steps-a.window)
        prev=np.zeros((n,len(steps)))
        valid=lower>0
        prev[:,valid]=csum[:,lower[valid]-1]
        rolling=(csum[:,steps-1]-prev)/(steps-lower)[None,:]
        axes2[panel].plot(steps,rolling.mean(axis=0),label=f'{alg} (n={n})')
    for axs in (axes,axes2):
        for ax,cond in zip(axs,('stationary','drift')):
            ax.set_title(cond); ax.set_xlabel('Online rounds'); ax.grid(alpha=.2)
            if ax.lines: ax.legend(fontsize=8)
            matching=[v for (c,alg),v in groups.items() if c==cond]
            if cond=='drift' and matching:
                interval=matching[0][0][2]['drift_every']
                maximum=max(len(r[0]) for v in matching for r in v)
                for t in range(interval,maximum,interval):ax.axvline(t,color='grey',ls=':',alpha=.4)
    axes[0].set_ylabel('0.93 × rounds − cumulative reward\n(paper-reference proxy)')
    axes2[0].set_ylabel(f'Accuracy over last ≤{a.window:,} rounds')
    fig.tight_layout(); fig2.tight_layout()
    fig.savefig(root/'regret.png',dpi=180); fig2.savefig(root/'accuracy.png',dpi=180)
    plt.close('all')
    with (root/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for row in rows:print(json.dumps(row))

if __name__=='__main__':main()
