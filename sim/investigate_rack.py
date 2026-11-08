"""Offline diagnostics on vendor recordings. Never advances simulation physics."""
import argparse
import importlib.util
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from scipy.optimize import least_squares


def records(root, phrase):
    matches=[]
    for path in sorted((root/'policy_records').glob('step_*.npy'),key=lambda p:int(p.stem.split('_')[1])):
        record=np.load(path,allow_pickle=True).item()
        if phrase in record['inputs/prompt'].lower():matches.append({'inputs':{k[len('inputs/'):]:v for k,v in record.items() if k.startswith('inputs/')}})
    return matches


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('rack',type=Path)
    parser.add_argument('baseline',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--policy-calls',type=int,nargs='+',default=[0,20])
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('official_image_tools',Path('.runtime/official-image-tools.py'))
    tools=importlib.util.module_from_spec(spec);spec.loader.exec_module(tools)
    native=records(args.rack,'server');positive=records(args.baseline,'open the middle drawer')
    assert len(native)==594 and positive
    folder=sorted((args.rack/'recordings').glob('ep_*'))[0]
    model=mujoco.MjModel.from_xml_path(str(folder/'model.xml'));data=mujoco.MjData(model)
    arrays=[]
    for path in sorted(folder.glob('state_*.npz')):
        with np.load(path,allow_pickle=False) as chunk:
            if chunk['states'].size:arrays.append(chunk['states'])
    states=np.concatenate(arrays)
    flag=mujoco.mjtState.mjSTATE_FULLPHYSICS
    handle=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'data_center_rack_b_1_r02_u21_handle_1')
    grip=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'gripper0_grip_site')
    renderer=mujoco.Renderer(model,256,256)
    option=mujoco.MjvOption();option.geomgroup[0]=0
    original_rgba=model.geom_rgba.copy()
    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or '' for i in range(model.ngeom)]
    report={'engine':mujoco.__version__,'frames':[],'reachability':[]}
    inputs={};conditions=[]
    def add(label,frame,external,wrist,state,prompt):
        index=len(conditions)
        inputs[f'{index}_external']=external;inputs[f'{index}_wrist']=wrist;inputs[f'{index}_state']=state
        conditions.append({'index':index,'condition':label,'frame':frame,'prompt':prompt})
    for phase,call in enumerate(args.policy_calls):
        # Policy request happens after ten settling actions, then every five actions.
        state_index=10+call*5
        mujoco.mj_setState(model,data,states[state_index],flag);mujoco.mj_forward(model,data)
        reference=native[call]['inputs'];pose=reference['observation/state']
        observation_offset=float(np.linalg.norm(data.site_xpos[grip]-pose[:3]));assert observation_offset<.002
        frame_report={'policy_call':call,'time_s':float(data.time),'cameras':{},'saved_state_observation_offset_m':observation_offset}
        views={}
        for camera,key in [('agentview','observation/image'),('robot0_eye_in_hand','observation/wrist_image')]:
            model.geom_rgba[:]=original_rgba
            renderer.update_scene(data,camera=camera,scene_option=option)
            rgb=renderer.render().copy()
            transforms={'identity':lambda x:x,'vertical':lambda x:x[::-1],'horizontal':lambda x:x[:,::-1],'rotate180':lambda x:x[::-1,::-1]}
            errors={name:float(np.abs(tools.resize_with_pad(np.ascontiguousarray(fn(rgb)),224,224).astype(float)-reference[key]).mean()) for name,fn in transforms.items()}
            orientation=min(errors,key=errors.get);orient=transforms[orientation]
            renderer.enable_segmentation_rendering();renderer.update_scene(data,camera=camera,scene_option=option)
            segment=renderer.render().copy();renderer.disable_segmentation_rendering()
            target=(segment[:,:,0]==handle)&(segment[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))
            frame_report['cameras'][camera]={'orientation_to_policy':orientation,'rgb_mean_absolute_errors':errors,'handle_pixels_256':int(target.sum()),'handle_pixels_224':int(np.asarray(Image.fromarray(orient(target).astype(np.uint8)).resize((224,224),Image.Resampling.NEAREST)).sum())}
            Image.fromarray(reference[key]).save(args.output/f'{phase}-{camera}-recorded.png')
            Image.fromarray(np.ascontiguousarray(orient(target)).astype(np.uint8)*255).save(args.output/f'{phase}-{camera}-handle-mask.png')
            for variant in ['rendered','single_rack','single_tray','table_visible']:
                model.geom_rgba[:]=original_rgba
                for i,name in enumerate(names):
                    if variant in ('single_rack','single_tray') and name.startswith('data_center_rack_'):
                        if not name.startswith('data_center_rack_b_1_') or (variant=='single_tray' and '_u' in name and 'r02_u21_' not in name):model.geom_rgba[i,3]=0
                    if variant=='table_visible' and (name=='table_visual' or name.startswith('table_leg')):model.geom_rgba[i,:]=1
                renderer.update_scene(data,camera=camera,scene_option=option)
                image=tools.convert_to_uint8(tools.resize_with_pad(np.ascontiguousarray(orient(renderer.render())),224,224))
                views[(variant,camera)]=image
                Image.fromarray(image).save(args.output/f'{phase}-{camera}-{variant}.png')
        report['frames'].append(frame_report)
        standard='Pull out the server with the cyan handle.'
        familiar='open the middle drawer of the cabinet'
        ext=reference['observation/image'];wrist=reference['observation/wrist_image']
        add('recorded_server',phase,ext,wrist,pose,standard)
        add('recorded_drawer',phase,ext,wrist,pose,familiar)
        add('zero_state',phase,ext,wrist,np.zeros_like(pose),standard)
        add('blank_external',phase,np.zeros_like(ext),wrist,pose,standard)
        add('blank_wrist',phase,ext,np.zeros_like(wrist),pose,standard)
        for variant in ['rendered','single_rack','single_tray','table_visible']:
            for label,prompt in [('server',standard),('drawer',familiar)]:
                add(f'{variant}_{label}',phase,views[(variant,'agentview')],views[(variant,'robot0_eye_in_hand')],pose,prompt)
        benchmark=positive[min(call,len(positive)-1)]['inputs']
        for label,prompt in [('familiar',benchmark['prompt']),('server',standard)]:
            add('benchmark_'+label,phase,benchmark['observation/image'],benchmark['observation/wrist_image'],benchmark['observation/state'],prompt)
        if phase==0:
            target=data.geom_xpos[handle].copy();initial_q=data.qpos[:7].copy();lower=model.jnt_range[:7,0]+1e-5;upper=model.jnt_range[:7,1]-1e-5
            for extension in [0,.3,.6]:
                desired=target+np.array([0,extension,0])
                def residual(q):
                    data.qpos[:7]=q;mujoco.mj_forward(model,data)
                    return data.site_xpos[grip]-desired
                candidates=[least_squares(residual,np.clip(start,lower,upper),bounds=(lower,upper),max_nfev=150) for start in [initial_q,(lower+upper)/2]]
                best=min(candidates,key=lambda x:np.linalg.norm(x.fun))
                report['reachability'].append({'extension_m':extension,'position_error_m':float(np.linalg.norm(best.fun)),'position_only':True,'collision_free_not_established':True})
                data.qpos[:7]=initial_q;mujoco.mj_forward(model,data)
    renderer.close()
    model.geom_rgba[:]=original_rgba
    # Geometric approach and gripper behavior in all three recorded trials.
    report['trajectories']=[]
    for folder in sorted((args.rack/'recordings').glob('ep_*')):
        chunks=[];actions=[]
        for path in sorted(folder.glob('state_*.npz')):
            with np.load(path,allow_pickle=True) as z:
                if z['states'].size:chunks.append(z['states'])
                actions.extend(x['actions'] for x in z['action_infos'])
        a=np.concatenate(chunks);distances=[];heights=[]
        for row in a:
            mujoco.mj_setState(model,data,row,flag);mujoco.mj_forward(model,data)
            distances.append(float(np.linalg.norm(data.site_xpos[grip]-data.geom_xpos[handle])));heights.append(float(data.site_xpos[grip,2]))
        actions=np.asarray(actions)[10:]
        nearest=int(np.argmin(distances))
        report['trajectories'].append({'episode':folder.name,'closest_handle_distance_m':min(distances),'closest_time_s':float(a[nearest,0]),'closing_command_fraction':float(np.mean(actions[:,6]>.5)),'downward_action_fraction':float(np.mean(actions[:,2]<-.02)),'minimum_gripper_height_m':min(heights)})
    np.savez_compressed(args.output/'inputs.npz',**inputs)
    (args.output/'conditions.json').write_text(json.dumps(conditions,indent=2))
    (args.output/'geometry.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
