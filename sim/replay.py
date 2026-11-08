import argparse
from pathlib import Path
import queue
import time
import mujoco
import numpy as np
from episodes import Episode

def main():
    p=argparse.ArgumentParser();p.add_argument('episode',type=Path);p.add_argument('--seconds',type=float);a=p.parse_args()
    a.episode=a.episode.resolve()
    paths=sorted(a.episode.parent.glob('*/manifest.json'));paths=[x.parent for x in paths]
    ep=Episode(a.episode);events=queue.SimpleQueue();index=0;paused=False;speed=1.;ep.seek(0)
    from mujoco import viewer
    with viewer.launch_passive(ep.model,ep.data,key_callback=events.put,show_left_ui=False,show_right_ui=False) as v:
        v.cam.lookat[:]=[0,-.5,1];v.cam.distance=5.3;v.cam.azimuth=125;v.cam.elevation=-23;v.opt.geomgroup[3]=0
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT]=False
        deadline=time.monotonic()+a.seconds if a.seconds else float('inf');last=time.monotonic();clock=float(ep.times[0])
        while v.is_running() and time.monotonic()<deadline:
            now=time.monotonic()
            if not paused:clock+=min(now-last,.1)*speed
            last=now
            while not events.empty():
                key=events.get()
                if key==32:paused=not paused
                elif key==82:clock=float(ep.times[0])
                elif key in (262,263):clock+=1 if key==262 else -1
                elif key in (44,46):
                    paused=True;index=ep.seek(index+(1 if key==46 else -1));clock=float(ep.times[index])
                elif key in (49,50,51,52):speed=[.25,.5,1.,2.][key-49]
                elif key in (72,70):
                    v.cam.type=mujoco.mjtCamera.mjCAMERA_FIXED if key==72 else mujoco.mjtCamera.mjCAMERA_FREE
                    if key==72:v.cam.fixedcamid=ep.model.camera('hardware').id
                elif key==73:
                    images=sorted(ep.path.glob('observation-*.npz'))
                    if images:
                        frame=min(len(images)-1,int(index/125))
                        with np.load(images[frame],allow_pickle=False) as obs:
                            v.set_images([(mujoco.MjrRect(0,0,224,224),obs['image']),(mujoco.MjrRect(224,0,224,224),obs['wrist'])])
                elif key in (78,80) and len(paths)>1:
                    offset=1 if key==78 else -1
                    target=paths[(paths.index(a.episode.resolve())+offset)%len(paths)] if a.episode.resolve() in paths else paths[0]
                    import subprocess,sys
                    command=[str(Path(__file__).with_name('replay.sh')),str(target)] if sys.platform=='darwin' else [sys.executable,__file__,str(target)]
                    subprocess.Popen(command);return
            clock=float(np.clip(clock,ep.times[0],ep.times[-1]));index=ep.seek(np.searchsorted(ep.times,clock,side='right')-1)
            if clock>=ep.times[-1]:paused=True
            v.set_texts((None,None,f'{ep.path.parent.name}/{ep.path.name} | {ep.meta["status"]}',f'{clock:.2f}/{ep.times[-1]:.2f} simulation seconds  {speed}x\nSpace play/pause | arrows seek | ,/. step | 1-4 speed\nR restart | H close-up | F orbit | I policy images | N/P episodes'))
            v.sync();time.sleep(.01)

if __name__=='__main__':main()
