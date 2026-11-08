"""Documented OpenPI client + LIBERO environment APIs; actions pass unchanged."""
import argparse,json,time,hashlib,shutil,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from PIL import Image
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.env_wrapper import ControlEnv
from robosuite.wrappers import DataCollectionWrapper
from robosuite.utils.transform_utils import quat2axisangle
from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy
import libero_task  # documented object-registration side effect
ROOT=Path(__file__).resolve().parent
TASK=ROOT/'tasks'/'pull_server.bddl'

def payload(obs):
    return {'observation/image':image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs['agentview_image'][::-1,::-1]),224,224)),
            'observation/wrist_image':image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1,::-1]),224,224)),
            'observation/state':np.concatenate((obs['robot0_eef_pos'],quat2axisangle(obs['robot0_eef_quat']),obs['robot0_gripper_qpos'])),
            'prompt':'Pull out the server with the cyan handle.'}

def scene_xml():
    # Documented arena customization: remove the table geometry, preserve the
    # named workspace anchors required by TableArena and the supplied camera rig.
    from libero.libero import get_libero_path
    source=Path(get_libero_path('assets'))/'scenes'/'libero_tabletop_base_style.xml'
    xml=ET.parse(str(source))
    for element in xml.findall('.//*[@file]'):
        element.set('file',str((source.parent/element.get('file')).resolve()))
    for geom in xml.findall('.//body[@name="table"]/geom'):
        geom.set('rgba','0 0 0 0');geom.set('contype','0');geom.set('conaffinity','0')
    destination=Path('/tmp/agverase-server-room.xml')
    xml.write(str(destination),encoding='unicode')
    return str(destination)

def make(render):
    if render:return OffScreenRenderEnv(bddl_file_name=str(TASK),scene_xml=scene_xml(),camera_heights=256,camera_widths=256)
    return ControlEnv(bddl_file_name=str(TASK),scene_xml=scene_xml(),use_camera_obs=False,has_renderer=False,has_offscreen_renderer=False)

def validate(output,render=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);env=make(render);obs=env.reset()
    report={'environment':type(env.env).__module__+'.'+type(env.env).__name__,'controller':type(env.robots[0].controller).__module__+'.'+type(env.robots[0].controller).__name__,'robot_model':type(env.robots[0].robot_model).__module__+'.'+type(env.robots[0].robot_model).__name__}
    assert report['controller'].startswith('robosuite.controllers.')
    camera=env.sim.model.camera_name2id('agentview')
    target=env.sim.data.geom_xpos[env.sim.model.geom_name2id('data_center_rack_b_1_r02_u21_handle_1')].copy()
    relative=env.sim.data.cam_xmat[camera].reshape(3,3).T.dot(target-env.sim.data.cam_xpos[camera])
    projected=relative[:2]/(-relative[2]*np.tan(np.deg2rad(env.sim.model.cam_fovy[camera])/2))
    report['target_camera_xy']=projected.tolist()
    if relative[2]>=0 or np.max(np.abs(projected))>.9:raise RuntimeError('Target outside usable supplied camera view')
    start=obs['robot0_eef_pos'].copy();peak=0
    for _ in range(200):
        obs,_,_,_=env.step([0,0,0,0,0,0,-1]);peak=max(peak,float(np.linalg.norm(obs['robot0_eef_pos']-start)))
    report['hold_drift_m']=peak
    if peak>.003:raise RuntimeError('Stock controller failed hold check')
    report['axis_responses']={}
    for axis in range(3):
        obs=env.reset()
        for _ in range(10):obs,_,_,_=env.step([0,0,0,0,0,0,-1])
        start=obs['robot0_eef_pos'].copy();action=np.array([0.,0,0,0,0,0,-1]);action[axis]=.1
        for _ in range(10):obs,_,_,_=env.step(action)
        report['axis_responses'][str(axis)]=(obs['robot0_eef_pos']-start).tolist()
        if obs['robot0_eef_pos'][axis]<=start[axis]:raise RuntimeError('Native axis direction check failed')
    if render:
        obs=env.reset();p=payload(obs)
        for key,label in [('observation/image','external'),('observation/wrist_image','wrist')]:Image.fromarray(p[key]).save(str(output/(label+'.png')))
        context=env.sim._render_context_offscreen
        report['visible_geoms']=int(context.scn.ngeom);report['render_capacity']=int(context.scn.maxgeom)
        if report['visible_geoms']>=report['render_capacity']:raise RuntimeError('Renderer capacity exceeded')
    (output/'validation.json').write_text(json.dumps(report,indent=2));env.close();print(json.dumps(report),flush=True)

def run(output,episodes=3,prompt=None,max_steps=None):
    if max_steps is not None and max_steps<1:raise ValueError('max_steps must be positive')
    output=Path(output);output.mkdir(parents=True,exist_ok=True);outer=make(True)
    env=DataCollectionWrapper(outer.env,str(output/'recordings'),collect_freq=1,flush_freq=100)
    client=WebsocketClientPolicy('127.0.0.1',8000);results=[]
    np.random.seed(7);outer.seed(7)
    try:
        for trial in range(episodes):
            obs=env.reset()
            for _ in range(10):obs,_,_,_=env.step([0,0,0,0,0,0,-1])
            joint=env.sim.model.get_joint_qpos_addr('data_center_rack_b_1_r02_u21_slide')
            initial_extension=float(env.sim.data.qpos[joint]);peak=initial_extension
            done=False;closest=float('inf');max_height=float(obs['robot0_eef_pos'][2])
            for step in range(min(env.horizon - 10,max_steps if max_steps is not None else env.horizon - 10)):
                if step%5==0:
                    p=payload(obs)
                    if prompt is not None:p['prompt']=prompt
                    chunk=np.asarray(client.infer(p)['actions'])
                    if chunk.shape[0]<5 or chunk.shape[1]!=7 or not np.isfinite(chunk).all():raise ValueError('Invalid policy action chunk')
                action=chunk[step%5]
                obs,_,done,_=env.step(action.tolist())
                peak=max(peak,float(env.sim.data.qpos[joint]))
                target=env.sim.data.geom_xpos[env.sim.model.geom_name2id('data_center_rack_b_1_r02_u21_handle_1')]
                closest=min(closest,float(np.linalg.norm(obs['robot0_eef_pos']-target)))
                max_height=max(max_height,float(obs['robot0_eef_pos'][2]))
                if done:break
            result={'trial':trial,'success':bool(done),'max_pull':peak,'initial_extension_m':initial_extension,'max_extension_change_m':max(0.,peak-initial_extension),'closest_handle_distance_m':closest,'max_gripper_height_m':max_height,'policy_steps':step+1};results.append(result);print(result,flush=True)
            (output/'results.json').write_text(json.dumps(results,indent=2))
    finally:
        env.close()
        for model_file in (output/'recordings').glob('*/model.xml'):
            shutil.copy2(model_file,model_file.with_name('model.original.xml'))
            xml=ET.parse(model_file);assets=model_file.parent/'assets';assets.mkdir(exist_ok=True)
            for element in xml.findall('.//*[@file]'):
                source=Path(element.get('file'));name=hashlib.sha256(source.read_bytes()).hexdigest()[:20]+source.suffix
                if not (assets/name).exists():shutil.copy2(source,assets/name)
                element.set('file','assets/'+name)
            compiler=xml.find('compiler')
            if compiler is not None:
                compiler.attrib.pop('meshdir',None);compiler.attrib.pop('texturedir',None)
            xml.write(str(model_file),encoding='unicode')
        import mujoco
        (output/'recording.json').write_text(json.dumps({'format':'robosuite.DataCollectionWrapper','mujoco':mujoco.__version__,'frequency_hz':20,'status':'completed' if len(results)==episodes else 'incomplete','trials':results},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['validate','camera-check','run']);p.add_argument('output');p.add_argument('--episodes',type=int,default=3);p.add_argument('--prompt');p.add_argument('--max-steps',type=int);a=p.parse_args()
    if a.command=='run':run(a.output,a.episodes,a.prompt,a.max_steps)
    else:validate(a.output,a.command=='camera-check')
