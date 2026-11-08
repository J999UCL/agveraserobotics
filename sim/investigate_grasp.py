"""Offline grasp-pose feasibility; never generates or executes a control trajectory."""
import argparse
import mujoco,numpy as np,json
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('episode',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args()
p=args.episode/'model.xml';m=mujoco.MjModel.from_xml_path(str(p));d=mujoco.MjData(m);s=np.load(sorted(p.parent.glob('state_*.npz'))[0],allow_pickle=False)['states'][10]
mujoco.mj_setState(m,d,s,mujoco.mjtState.mjSTATE_FULLPHYSICS);mujoco.mj_forward(m,d)
g=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'data_center_rack_b_1_r02_u21_handle_1');sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'gripper0_grip_site');j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'data_center_rack_b_1_r02_u21_slide');addr=m.jnt_qposadr[j]
lo=m.jnt_range[:7,0]+1e-4;hi=m.jnt_range[:7,1]-1e-4;initial=d.qpos[:7].copy();names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or '' for i in range(m.ngeom)]
# Finger approach into the rack (-Y), jaw opening across the vertical handle (X).
desired_rotation=np.array([[0,1,0],[0,0,-1],[-1,0,0]],float);reports=[]
for extension in [0,.3,.6]:
 d.qpos[addr]=extension;mujoco.mj_forward(m,d);desired=d.geom_xpos[g].copy()+[0,.005,0]
 def residual(q):
  d.qpos[:7]=q;mujoco.mj_forward(m,d)
  return np.r_[d.site_xpos[sid]-desired,.2*Rotation.from_matrix(desired_rotation@d.site_xmat[sid].reshape(3,3).T).as_rotvec()]
 best=None;best_score=float('inf')
 for start in [initial,(lo+hi)/2]+[np.random.default_rng(seed).uniform(lo,hi) for seed in range(4)]:
  sol=least_squares(residual,np.clip(start,lo,hi),bounds=(lo,hi),max_nfev=180);error=residual(sol.x)
  hits=[]
  for c in d.contact:
   a,b=names[c.geom1],names[c.geom2]
   if ((a.startswith(('robot0_','gripper0_')) and b.startswith('data_center_rack_')) or (b.startswith(('robot0_','gripper0_')) and a.startswith('data_center_rack_'))) and c.dist<-.0001:hits.append({'geoms':[a,b],'penetration_m':float(-c.dist)})
  score=float(np.linalg.norm(error))+sum(h['penetration_m'] for h in hits)
  if score<best_score:
   best_score=score;best={'extension_m':extension,'position_error_m':float(np.linalg.norm(error[:3])),'orientation_error_rad':float(np.linalg.norm(error[3:])/.2),'penetrations':hits,'arm_qpos':sol.x.tolist()}
 reports.append(best)
args.output.write_text(json.dumps(reports,indent=2));print(json.dumps(reports,indent=2))
