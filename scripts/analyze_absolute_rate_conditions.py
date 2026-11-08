#!/usr/bin/env python3
"""Across-champion correlations at matched noise conditions, with seed-level inference."""
import csv
import json
from pathlib import Path
import numpy as np
from analyze_absolute_rate_fitness import OUT, NAMES, rho


def standardized_ranks(values):
    values = np.round(values, 12)
    # Each column is a condition. Rank champions separately within each column.
    ranks = (values[:, None, :] > values[None, :, :]).sum(axis=1)
    ranks = ranks + ((values[:, None, :] == values[None, :, :]).sum(axis=1) - 1) / 2
    ranks -= ranks.mean(axis=0)
    norms = np.linalg.norm(ranks, axis=0)
    return np.divide(ranks, norms, out=np.zeros_like(ranks), where=norms != 0), norms != 0


def statistic(fitness, rates):
    x, valid_x = standardized_ranks(fitness)
    y, valid_y = standardized_ranks(rates)
    valid = valid_x & valid_y
    return (float(np.mean(np.sum(x[:, valid]*y[:, valid], axis=0))) if valid.any() else np.nan,
            x, y, valid)


def main():
    agents = [json.loads(p.read_text()) for p in sorted(OUT.glob('*.json'))]
    assert len(agents) == 100
    rng = np.random.default_rng(43)
    results = []
    pair_rows = []
    for env in NAMES:
        selected = [a for a in agents if a['agent']['environment'] == env]
        n = len(selected)
        condition_keys = sorted((c['observation'], c['alpha']) for c in selected[0]['conditions'])
        assert condition_keys
        expected_dims = dict(mountaincar_continuous=4,pendulum=3,acrobot=4,reacher=10,halfcheetah=17)
        assert len(condition_keys) == expected_dims[env]*6
        ordered = []
        for a in selected:
            lookup = {(c['observation'],c['alpha']):c for c in a['conditions']}
            assert sorted(lookup) == condition_keys
            ordered.append([lookup[k] for k in condition_keys])
            for c in [dict(observation=-1,alpha=0,**a['baseline'])]+a['conditions']:
                pair_rows.append(dict(environment=env,seed=a['agent']['seed'],observation=c['observation'],
                    alpha=c['alpha'],fitness=float(np.mean(c['fitness'])),
                    combined_final_rate=float(np.mean(c['final_rates'])),
                    combined_mean_rate=float(np.mean(c['mean_rates'])),
                    **{name:c['final_rates'][i] for i,name in enumerate(['swap','delete','add','mutate'])}))
        fitness = np.array([[np.mean(c['fitness']) for c in cs] for cs in ordered])
        for stage in ['final_rates','mean_rates']:
            all_rates=np.array([[c[stage] for c in cs] for cs in ordered])
            for i,metric in enumerate(['combined','swap','delete','add','mutate']):
                rates=all_rates.mean(axis=2) if i==0 else all_rates[:,:,i-1]
                observed,x,y,valid=statistic(fitness,rates)
                if not valid.any():
                    continue
                permutations=[np.mean(np.sum(x[:,valid]*y[rng.permutation(n)][:,valid],axis=0)) for _ in range(10000)]
                p=(1+np.sum(np.abs(permutations)>=abs(observed)-1e-12))/10001
                bootstrap=[]
                for _ in range(1000):
                    indices=rng.integers(n,size=n)
                    bootstrap.append(statistic(fitness[indices],rates[indices])[0])
                low,high=np.nanquantile(bootstrap,[.025,.975])
                results.append(dict(environment=env,stage=stage,metric=metric,n=n,conditions=len(condition_keys),
                    mean_condition_rho=observed,ci_low=low,ci_high=high,p_two_sided=p))
    for stage in ['final_rates','mean_rates']:
        primary=[r for r in results if r['metric']=='combined' and r['stage']==stage]
        previous=0
        for i,r in enumerate(sorted(primary,key=lambda r:r['p_two_sided'])):
            previous=max(previous,min(1,(len(primary)-i)*r['p_two_sided']))
            r['p_holm_5_environments']=previous
    for filename,rows in [('matched_condition_correlations.csv',results),('fitness_actual_rate_pairs.csv',pair_rows)]:
        fields=list(rows[0])
        if filename.startswith('matched') and 'p_holm_5_environments' not in fields:fields.append('p_holm_5_environments')
        with (OUT/filename).open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    for r in results:
        if r['stage']=='final_rates' and r['metric']=='combined':print(r)
    half=[a for a in agents if a['agent']['environment']=='halfcheetah' and a['agent']['seed']!=19]
    print('Half-Cheetah clean rho excluding variable seed 19:',rho(
        np.array([np.mean(a['baseline']['fitness']) for a in half]),
        np.array([np.mean(a['baseline']['final_rates']) for a in half])))


if __name__ == '__main__':
    main()
