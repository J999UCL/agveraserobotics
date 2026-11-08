"""Portable, chunked MuJoCo recordings. No inference or physics during playback."""
import hashlib
import json
from pathlib import Path
import shutil
import time
import mujoco
import numpy as np

FLAG=int(mujoco.mjtState.mjSTATE_INTEGRATION)

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class Episode:
    def __init__(self,path):
        self.path=Path(path);self.meta=json.loads((self.path/'manifest.json').read_text())
        self.portable=self.meta.get('format')=='native-qpos-v1'
        expected=self.meta.get('replay_mujoco',self.meta['mujoco']) if self.portable else self.meta['mujoco']
        if self.meta['version']!=1 or expected!=mujoco.__version__:raise ValueError('Replay needs MuJoCo '+expected)
        scene=(self.path/self.meta['scene']).resolve()
        if digest(scene)!=self.meta['scene_hash']:raise ValueError('Scene hash mismatch')
        for p,h in json.loads((scene.parent/'assets-hashes.json').read_text()).items():
            if digest(scene.parent/p)!=h:raise ValueError(f'Asset mismatch: {p}')
        self.model=mujoco.MjModel.from_xml_path(str(scene));self.data=mujoco.MjData(self.model)
        chunks=[]
        for chunk in self.meta.get('chunks',[]):
            p=self.path/chunk['file']
            if digest(p)!=chunk['sha256']:raise ValueError('Recording chunk corrupted')
            with np.load(p,allow_pickle=False) as a:chunks.append(a['states'])
        if not chunks:raise ValueError('No committed recording frames')
        self.states=np.concatenate(chunks);self.times=self.states[:,0]
        if self.portable:
            layout=self.meta['pose_layout']
            if (self.model.nq,self.model.nv,self.model.nu)!=(layout['nq'],layout['nv'],layout['nu']):raise ValueError('Native pose dimensions changed')
            expected_size=1+self.model.nq+self.model.nv+self.model.nu
        else:expected_size=mujoco.mj_stateSize(self.model,self.meta['state_flag'])
        if self.states.shape[1]!=expected_size:raise ValueError('State layout mismatch')
    def seek(self,index):
        index=int(np.clip(index,0,len(self.states)-1))
        if self.portable:
            row=self.states[index];nq=self.model.nq;nv=self.model.nv
            self.data.time=row[0];self.data.qpos[:]=row[1:1+nq];self.data.qvel[:]=row[1+nq:1+nq+nv];self.data.ctrl[:]=row[1+nq+nv:]
        else:mujoco.mj_setState(self.model,self.data,self.states[index],self.meta['state_flag'])
        mujoco.mj_forward(self.model,self.data)
        return index

def hash_assets(root):
    root=Path(root)
    (root/'assets-hashes.json').write_text(json.dumps({str(p.relative_to(root)):digest(p) for p in sorted((root/'assets').rglob('*')) if p.is_file()},indent=2))
