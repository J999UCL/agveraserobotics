"""Read-only playback of vendor DataCollectionWrapper states, same MuJoCo version."""
import argparse,hashlib,json,queue,time
from pathlib import Path
import mujoco
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('episode',type=Path);a=p.parse_args();root=a.episode
 metadata=json.loads((root.parent.parent/'recording.json').read_text())
 if metadata['mujoco']!=mujoco.__version__:raise ValueError('Replay needs MuJoCo '+metadata['mujoco'])
 for asset in (root/'assets').iterdir():
  if hashlib.sha256(asset.read_bytes()).hexdigest()[:20]!=asset.stem:raise ValueError('Asset download incomplete or corrupted: '+asset.name)
 model=mujoco.MjModel.from_xml_path(str(root/'model.xml'));data=mujoco.MjData(model)
 arrays=[]
 for f in sorted(root.glob('state_*.npz')):
  with np.load(f,allow_pickle=False) as z:
   if z['states'].size:arrays.append(z['states'])
 if not arrays:raise ValueError('No saved states')
 states=np.concatenate(arrays);flag=mujoco.mjtState.mjSTATE_FULLPHYSICS
 if states.shape[1]!=mujoco.mj_stateSize(model,flag):raise ValueError('State layout differs; use the recorded MuJoCo version')
 events=queue.SimpleQueue();paused=False;speed=1.;index=0;clock=float(states[0,0]);last=time.monotonic()
 from mujoco import viewer
 with viewer.launch_passive(model,data,key_callback=events.put) as v:
  v.cam.lookat[:]=[0,-.6,1.];v.cam.distance=4.;v.cam.azimuth=120;v.cam.elevation=-20
  print('Space pause | arrows seek | R restart | 1-4 speed | mouse orbit',flush=True)
  while v.is_running():
   now=time.monotonic()
   if not paused:clock+=min(now-last,.1)*speed
   last=now
   while not events.empty():
    key=events.get()
    if key==32:paused=not paused
    elif key in (262,263):clock+=1 if key==262 else -1
    elif key==82:clock=states[0,0];paused=False
    elif key in (49,50,51,52):speed=[.25,.5,1.,2.][key-49]
   clock=float(np.clip(clock,states[0,0],states[-1,0]));index=max(0,np.searchsorted(states[:,0],clock,side='right')-1)
   mujoco.mj_setState(model,data,states[index],flag);mujoco.mj_forward(model,data);v.sync()
   if index==len(states)-1:paused=True
   time.sleep(.01)
if __name__=='__main__':main()
