"""Read-only geometry and contact audit of standard vendor state recordings."""
import argparse,json
from pathlib import Path
import mujoco
import numpy as np

PHYSICAL_FIELDS=['body_mass','body_inertia','body_pos','body_quat','geom_type','geom_size','geom_pos','geom_quat','geom_contype','geom_conaffinity','geom_friction','geom_solref','geom_solimp','jnt_pos','jnt_axis','jnt_range','dof_damping','dof_frictionloss','actuator_gear','actuator_gainprm','actuator_biasprm','actuator_ctrlrange','mesh_vert','mesh_face']

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);p.add_argument('--cases',nargs='+',default=['original','wording','clutter_wording']);a=p.parse_args()
    reference=mujoco.MjModel.from_xml_path(str(next((a.root/a.cases[0]/'recordings').glob('*/model.xml'))))
    report={'engine':mujoco.__version__,'physical_fields_checked':PHYSICAL_FIELDS,'cases':{}};traces={}
    for case in a.cases:
        episodes=[]
        for trial,path in enumerate(sorted((a.root/case/'recordings').glob('*/model.xml'))):
            m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m)
            differences=[field for field in PHYSICAL_FIELDS if not np.array_equal(getattr(reference,field),getattr(m,field))]
            if differences:raise ValueError('Physical model changed: '+str(differences))
            arrays=[]
            for chunk in sorted(path.parent.glob('state_*.npz')):
                with np.load(chunk,allow_pickle=False) as z:
                    if z['states'].size:arrays.append(z['states'])
            states=np.concatenate(arrays);flag=mujoco.mjtState.mjSTATE_FULLPHYSICS
            assert states.shape[1]==mujoco.mj_stateSize(m,flag)
            sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'gripper0_grip_site')
            handles=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'data_center_rack_b_1_r02_u21_handle_'+str(sign)) for sign in [-1,1]]
            joint=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'data_center_rack_b_1_r02_u21_slide');qaddr=m.jnt_qposadr[joint]
            names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or '' for i in range(m.ngeom)]
            robot=np.array([n.startswith(('robot0_','gripper0_')) for n in names]);gripper=np.array([n.startswith('gripper0_') for n in names]);target=np.array([n.startswith('data_center_rack_b_1_r02_u21_') for n in names])
            closest=float('inf');grip_contacts=0;robot_contacts=0;xyz=[]
            for row in states:
                mujoco.mj_setState(m,d,row,flag);mujoco.mj_forward(m,d);xyz.append(d.site_xpos[sid].copy())
                closest=min(closest,min(float(np.linalg.norm(d.site_xpos[sid]-d.geom_xpos[h])) for h in handles))
                touch_gripper=False;touch_robot=False
                for c in d.contact:
                    i,j=c.geom1,c.geom2
                    if c.dist>0:continue
                    if (gripper[i] and target[j]) or (gripper[j] and target[i]):touch_gripper=True
                    if (robot[i] and target[j]) or (robot[j] and target[i]):touch_robot=True
                grip_contacts+=int(touch_gripper);robot_contacts+=int(touch_robot)
            initial=float(states[10,1+qaddr]);peak=float(states[:,1+qaddr].max())
            episodes.append({'trial':trial,'episode':path.parent.name,'frames':len(states),'closest_either_handle_distance_m':closest,'target_gripper_contact_frames':grip_contacts,'target_robot_contact_frames':robot_contacts,'initial_extension_m':initial,'maximum_extension_m':peak,'extension_change_from_initial_m':max(0.,peak-initial),'physical_fields_identical':True,'visual_rgba_changed':not np.array_equal(reference.geom_rgba,m.geom_rgba)})
            traces[f'{case}_{trial}_eef']=np.array(xyz);traces[f'{case}_{trial}_time']=states[:,0]
        report['cases'][case]=episodes
    a.output.write_text(json.dumps(report,indent=2));np.savez_compressed(a.output.with_suffix('.npz'),**traces);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
