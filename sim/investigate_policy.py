"""Matched-noise input experiments through OpenPI's public Policy.infer API."""
import argparse
import json
from pathlib import Path
import time
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('inputs',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args()
    from openpi.training import config
    from openpi.policies import policy_config
    cfg=config.get_config('pi05_libero')
    policy=policy_config.create_trained_policy(cfg,'/data/weights/openpi-assets/checkpoints/pi05_libero')
    conditions=json.loads((args.inputs/'conditions.json').read_text());args.output.mkdir(parents=True,exist_ok=True)
    payloads=[]
    with np.load(args.inputs/'inputs.npz',allow_pickle=False) as bundle:
        for c in conditions:
            n=c['index'];payloads.append({'observation/image':bundle[f'{n}_external'],'observation/wrist_image':bundle[f'{n}_wrist'],'observation/state':bundle[f'{n}_state'],'prompt':c['prompt']})
    (args.output/'conditions.json').write_text(json.dumps(conditions,indent=2))
    start=time.monotonic()
    for seed in range(8):
        noise=np.random.default_rng(seed).standard_normal((cfg.model.action_horizon,cfg.model.action_dim)).astype(np.float32)
        actions=np.empty((len(conditions),cfg.model.action_horizon,7))
        order=np.random.default_rng(seed+100).permutation(len(conditions))
        for index in order:
            actions[index]=policy.infer(payloads[index],noise=noise)['actions']
            if not np.isfinite(actions[index]).all():raise ValueError('Nonfinite actions')
        np.savez_compressed(args.output/f'seed-{seed}.npz',actions=actions,noise=noise)
        print(json.dumps({'seed':seed,'conditions':len(conditions),'elapsed_s':time.monotonic()-start}),flush=True)
    (args.output/'status.json').write_text(json.dumps({'status':'completed','seeds':8,'conditions':len(conditions),'revision':'215abfb217dbac7d5f1273282331b9b1866c0479','method':'public Policy.infer with paired explicit noise'}))

if __name__=='__main__':main()
