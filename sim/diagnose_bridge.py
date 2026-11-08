"""Audit recorded policy commands against actual motion; never runs inference."""
import argparse,json
from pathlib import Path
import numpy as np
from episodes import Episode

def audit(path):
 e=Episode(path);m=e.model;sid=m.site('grip').id
 rows=[]
 for line in (e.path/'events.jsonl').read_text().splitlines():
  event=json.loads(line)
  if 'executed' not in event:continue
  a=np.array(event['executed'],dtype=float)
  if 'policy_action' not in event and e.meta.get('policy_frame'):
   R=np.array(e.meta['policy_frame']['rotation']);a[:3]=R@a[:3];a[3:6]=R@a[3:6]
  t=float(event['time']);i=np.searchsorted(e.times,t,side='left');j=np.searchsorted(e.times,t+.05,side='left')
  e.seek(i);p=e.data.site_xpos[sid].copy();command=np.clip(a[:3],-1,1)*.05
  e.seek(j);actual=e.data.site_xpos[sid]-p
  rows.append(dict(time=t,requested_delta=command.tolist(),actual_delta=actual.tolist(),tracking_error=float(np.linalg.norm(actual-command)),height=float(e.data.site_xpos[sid,2])))
 errors=np.array([r['tracking_error'] for r in rows]);commands=np.array([r['requested_delta'] for r in rows]);motion=np.array([r['actual_delta'] for r in rows])
 early=errors[:100]
 summary=dict(episode=str(e.path),steps=len(rows),summed_requested_delta_m=commands.sum(0).tolist(),summed_actual_delta_m=motion.sum(0).tolist(),early_median_tracking_error_m=float(np.median(early)),overall_median_tracking_error_m=float(np.median(errors)),p95_tracking_error_m=float(np.percentile(errors,95)),downward_command_fraction=float(np.mean(commands[:,2]<-.001)),height_min_m=min(r['height'] for r in rows),height_max_m=max(r['height'] for r in rows))
 return summary,rows

def main():
 p=argparse.ArgumentParser();p.add_argument('episode',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
 summary,rows=audit(a.episode);print(json.dumps(summary,indent=2))
 if a.output:a.output.mkdir(parents=True,exist_ok=True);(a.output/'bridge-audit.json').write_text(json.dumps(dict(summary=summary,steps=rows),indent=2))
if __name__=='__main__':main()
