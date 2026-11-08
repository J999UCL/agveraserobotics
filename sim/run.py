"""Interactive rack servicing scene, preview renderer and physics checks."""
import argparse
from pathlib import Path
import queue
import time
import mujoco
import numpy as np
from server_hardware import COUNT, TRAVEL

ROOT=Path(__file__).resolve().parent

def tray_name(rack,slot):return f'r{rack:02}_u{slot*2+1:02}_slide'

def load():
    m=mujoco.MjModel.from_xml_path(str(ROOT/'datacentre.xml'))
    d=mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m,d,0)
    mujoco.mj_forward(m,d)
    return m,d

def drive(m,d,targets):
    d.qfrc_applied[:]=0
    for name,target in targets.items():
        j=m.joint(name);q=int(j.qposadr[0]);v=int(j.dofadr[0])
        # Bounded rail service force; the arm itself is never teleported or commanded.
        d.qfrc_applied[v]=np.clip(900*(target-d.qpos[q])-130*d.qvel[v],-100,100)
    mujoco.mj_step(m,d)

def settled(m,d,targets,tolerance=.012):
    return all(abs(float(d.joint(name).qpos[0])-target)<tolerance for name,target in targets.items())

def validate(m,d,exercise=False):
    start=d.qpos.copy()
    for _ in range(5000):drive(m,d,{})
    assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
    assert np.max(np.abs(d.qpos-start))<.1,'Unexpected home-pose drift'
    assert not any(w.number for w in d.warning),'MuJoCo physics warning'
    print(f'PASS: 10s at home; {4*COUNT} sliding servers, {m.nbody} bodies, {m.ngeom} geoms; no warnings')
    if exercise:
        # Validate every rail with the parked arm's collision geometry disabled.
        # This checks rack clearances independently of robot obstruction.
        cart_id=m.body('service_cart').id
        robot_geoms=[]
        for i in range(m.ngeom):
            body=int(m.geom_bodyid[i])
            while body and body!=cart_id:body=int(m.body_parentid[body])
            if body==cart_id:robot_geoms.append(i)
        original_type=m.geom_contype.copy();original_affinity=m.geom_conaffinity.copy()
        for i in robot_geoms:m.geom_contype[i]=0;m.geom_conaffinity[i]=0
        targets={tray_name(r,s):TRAVEL-.02 for r in range(1,5) for s in range(COUNT)}
        for _ in range(3000):drive(m,d,targets)
        assert settled(m,d,targets), 'One or more trays failed to extend'
        targets={name:0 for name in targets}
        for _ in range(3000):drive(m,d,targets)
        assert settled(m,d,targets), 'One or more trays failed to close'
        assert not any(w.number for w in d.warning),'MuJoCo warning during travel'
        m.geom_contype[:]=original_type;m.geom_conaffinity[:]=original_affinity
        print('PASS: all 80 rails extended to 0.70m and returned to closed under bounded force (parked arm/cart contacts disabled for rail clearance check)')
    mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)

SHOWCASE={tray_name(4,5):.46,tray_name(4,10):.70,tray_name(4,15):.36}

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--render',type=Path)
    p.add_argument('--camera',choices=['overview','front','service','hardware'],default='overview')
    p.add_argument('--check',action='store_true')
    p.add_argument('--check-trays',action='store_true')
    p.add_argument('--showcase',action='store_true',help='Open three right-rack servers to expose their internals')
    p.add_argument('--seconds',type=float)
    args=p.parse_args()
    m,d=load()
    if args.check or args.check_trays:
        validate(m,d,args.check_trays)
        if not args.render:return
    targets=dict(SHOWCASE) if args.showcase else {}
    if args.render:
        from PIL import Image
        for _ in range(2200):drive(m,d,targets)
        assert settled(m,d,targets),'Showcase trays obstructed'
        args.render.parent.mkdir(parents=True,exist_ok=True)
        with mujoco.Renderer(m,height=1200,width=1600) as renderer:
            opt=mujoco.MjvOption();opt.geomgroup[3]=0
            renderer.update_scene(d,camera=args.camera,scene_option=opt)
            Image.fromarray(renderer.render()).save(args.render)
        print(f'Saved {args.render}')
        return
    # Key callbacks queue events; all model/state updates happen on this thread.
    from mujoco import viewer
    events=queue.SimpleQueue()
    selected_rack=4;selected_slot=10
    print('1–4: select rack | Up/Down: select 2U server | O: extend | C: close | M: release service force | V: showcase | R: reset | H: hardware camera | F: orbit camera',flush=True)
    previous_handles=[]
    def announce():
        for gid in previous_handles:m.geom_matid[gid]=m.material('white_trim').id
        previous_handles.clear()
        for side in [-1,1]:
            gid=m.geom(f'r{selected_rack:02}_u{selected_slot*2+1:02}_handle_{side}').id
            m.geom_matid[gid]=m.material('cyan').id
            previous_handles.append(gid)
        print(f'Selected R{selected_rack:02} U{selected_slot*2+1:02}–U{selected_slot*2+2:02} (cyan handles)',flush=True)
    announce()
    with viewer.launch_passive(m,d,key_callback=events.put,show_left_ui=False,show_right_ui=False) as v:
        with v.lock():
            v.cam.type=mujoco.mjtCamera.mjCAMERA_FREE
            v.cam.lookat[:]=[0,-.3,1.0];v.cam.distance=5.3;v.cam.azimuth=125;v.cam.elevation=-23
            if args.camera!='overview':
                v.cam.type=mujoco.mjtCamera.mjCAMERA_FIXED
                v.cam.fixedcamid=m.camera(args.camera).id
            v.opt.geomgroup[3]=0
        end=time.monotonic()+args.seconds if args.seconds else float('inf')
        while v.is_running() and time.monotonic()<end:
            started=time.monotonic()
            while not events.empty():
                key=events.get()
                name=tray_name(selected_rack,selected_slot)
                if 49<=key<=52:selected_rack=key-48;announce()
                elif key in [264,265]:selected_slot=int(np.clip(selected_slot+(1 if key==265 else -1),0,COUNT-1));announce()
                elif key==79:targets[name]=TRAVEL-.02;print('Extending',name,flush=True)
                elif key==67:targets[name]=0;print('Closing',name,flush=True)
                elif key==77:targets.pop(name,None);print('Passive rail:',name,flush=True)
                elif key==86:targets.update(SHOWCASE)
                elif key==82:
                    targets.clear();mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
                elif key in [72,70]:
                    with v.lock():
                        v.cam.type=mujoco.mjtCamera.mjCAMERA_FIXED if key==72 else mujoco.mjtCamera.mjCAMERA_FREE
                        if key==72:v.cam.fixedcamid=m.camera('hardware').id
            for _ in range(10):drive(m,d,targets)
            v.sync()
            time.sleep(max(0,.02-(time.monotonic()-started)))

if __name__=='__main__':main()
