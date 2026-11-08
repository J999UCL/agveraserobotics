"""Replay identical recorded commands through unmodified vendor environments."""
import json
from pathlib import Path
import numpy as np
from robosuite.wrappers import DataCollectionWrapper
from libero_client import make


def replay(actions,wrapped):
    np.random.seed(7)
    outer=make(False);outer.seed(7)
    env=DataCollectionWrapper(outer.env,'/tmp/agverase-wrapper-check',collect_freq=1,flush_freq=100) if wrapped else outer.env
    obs=env.reset();states=[env.sim.get_state().flatten()];positions=[obs['robot0_eef_pos'].copy()]
    try:
        for action in actions:
            obs,_,_,_=env.step(action.tolist())
            states.append(env.sim.get_state().flatten());positions.append(obs['robot0_eef_pos'].copy())
    finally:env.close()
    return np.array(states),np.array(positions)

if __name__=='__main__':
    actions=np.load('/opt/experiment-inputs/replay-actions.npy',allow_pickle=False)
    plain,p0=replay(actions,False);recorded,p1=replay(actions,True)
    result={'steps':len(actions),'max_eef_difference_m':float(np.max(np.linalg.norm(p0-p1,axis=1))),'max_flattened_state_difference':float(np.max(np.abs(plain-recorded)))}
    Path('/tmp/wrapper-result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    if result['max_eef_difference_m']>.001:raise RuntimeError('Recording wrapper materially changed trajectory')
