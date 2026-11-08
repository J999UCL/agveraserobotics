"""Summarize paired public-API inference probes; these are not task success trials."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('inference',type=Path);a=p.parse_args()
    conditions=json.loads((a.inference/'conditions.json').read_text())
    actions=np.stack([np.load(f,allow_pickle=False)['actions'] for f in sorted(a.inference.glob('seed-*.npz'))])
    with np.load(a.root/'inputs.npz',allow_pickle=False) as z:states={c['frame']:z[f"{c['index']}_state"] for c in conditions if c['condition']=='recorded_server'}
    target=np.array([-.225,-.33095537,1.049])
    summaries=[];effects=[]
    for c in conditions:
        index=c['index'];plans=actions[:,index,:5,:].mean(axis=1);delta=target-states[c['frame']][:3];unit=delta/np.linalg.norm(delta)
        dots=plans[:,:3]@unit;cos=dots/np.maximum(1e-9,np.linalg.norm(plans[:,:3],axis=1))
        s={'frame':c['frame'],'condition':c['condition'],'mean_dxyz':plans[:,:3].mean(axis=0).tolist(),'mean_direction_cosine_to_rack_handle':float(cos.mean()),'seeds_toward_rack_handle':int((dots>0).sum()),'seeds':len(actions),'mean_gripper_command':float(plans[:,6].mean())}
        if c['condition'].startswith('benchmark'):s['rack_direction_not_applicable']=True
        summaries.append(s)
    for frame in sorted(states):
        lookup={c['condition']:c['index'] for c in conditions if c['frame']==frame}
        for control,changed in [('recorded_server','zero_state'),('recorded_server','recorded_drawer'),('rendered_server','single_rack_server'),('rendered_server','single_tray_server'),('rendered_server','table_visible_server'),('rendered_drawer','single_rack_drawer'),('rendered_drawer','single_tray_drawer'),('rendered_drawer','table_visible_drawer'),('benchmark_familiar','benchmark_server')]:
            difference=actions[:,lookup[changed]]-actions[:,lookup[control]]
            effects.append({'frame':frame,'control':control,'intervention':changed,'max_abs_action_change':float(np.max(np.abs(difference))),'mean_abs_action_change':float(np.mean(np.abs(difference)))})
    report={'method':'8 paired noise seeds, first five actions of each ten-action prediction; no actions executed','conditions':summaries,'effects':effects}
    (a.root/'inference-summary.json').write_text(json.dumps(report,indent=2))
    for s in summaries:
        print(s['frame'],s['condition'],'toward',str(s['seeds_toward_rack_handle'])+'/8','cos',round(s['mean_direction_cosine_to_rack_handle'],3),'dxyz',np.round(s['mean_dxyz'],3),'grip',round(s['mean_gripper_command'],3))
    print('STATE EFFECT',[e['max_abs_action_change'] for e in effects if e['intervention']=='zero_state'])

if __name__=='__main__':main()
