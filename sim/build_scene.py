"""Convert the vendored GLB and compose a reproducible MuJoCo maintenance bay."""
from pathlib import Path
import copy
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
from server_hardware import hardware_assets, populate, COUNT

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'assets' / 'converted'
OUT.mkdir(exist_ok=True)

def add(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k,v in attrs.items()})

def box(parent, name, pos, size, material, **kw):
    return add(parent, 'geom', name=name, type='box', pos=pos, size=size, material=material, **kw)

# Service base geometry (metres). Shared with robot/arm/base.py through the compiled model.
RAIL_Y=-.90              # rail centreline, in front of the rack fronts at y=-.47
RAIL_RANGE=(-1.5,1.5)    # carriage travel along the aisle
HOME_X=1.46              # parked beside rack 4, clear of every tray
LIFT_LOW=.22             # arm base height with the lift fully down
LIFT_TRAVEL=1.55         # lift stroke: arm base from .22 to 1.77 m
GRIP_KP=4000             # gripper servo, N/m
GRIP_PAST=.01            # 'closed' aims 10 mm past shut, so a held 14 mm handle gets ~45 N (real Franka: up to 70 N)
BASE_KP=20000            # rail / lift servo stiffness, N/m
RAIL_KV=2200             # ~critical damping for the ~60 kg carriage
LIFT_KV=1400             # ~critical damping for the ~25 kg lift and arm

def font(size):
    for path in ['/System/Library/Fonts/Helvetica.ttc','C:/Windows/Fonts/arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if Path(path).exists():return ImageFont.truetype(path,size)
    return ImageFont.load_default(size)

def label(text, name):
    im=Image.new('RGB',(512,128),'#15212e'); d=ImageDraw.Draw(im)
    font_=font(48)
    d.rectangle((0,0,12,128),fill='#46cfdd')
    d.text((36,34),text,font=font_,fill='#e4f4f6')
    im.save(OUT / f'{name}.png')

root=ET.Element('mujoco',model='Agverase data centre')
add(root,'compiler',angle='radian',autolimits='true')
# Elliptic friction cone + high impratio: MuJoCo's recommended setting against grasp slip.
add(root,'option',timestep='0.002',integrator='implicitfast',cone='elliptic',impratio='10')
visual=add(root,'visual')
add(visual,'global',offwidth='1600',offheight='1200')
add(visual,'quality',shadowsize='4096',offsamples='4')
add(visual,'headlight',ambient='0.22 0.22 0.25',diffuse='0.25 0.25 0.25',specular='0.1 0.1 0.1')
add(visual,'map',znear='0.01',zfar='30')
add(root,'statistic',center='0 -0.5 0.95',extent='3.7')
assets=add(root,'asset')
for name,rgba in {'graphite':'.055 .07 .09 1','metal':'.32 .38 .43 1','yellow':'.94 .64 .035 1','rubber':'.025 .029 .035 1','floor_a':'.49 .54 .58 1','floor_b':'.53 .58 .62 1','line':'.98 .71 .10 1','cyan':'.07 .65 .76 1','white_trim':'.72 .78 .82 1'}.items():
    add(assets,'material',name=name,rgba=rgba,specular='.25',shininess='.3')
add(assets,'texture',name='sky',type='skybox',builtin='gradient',rgb1='.24 .30 .38',rgb2='.65 .71 .77',width='512',height='3072')
scene=trimesh.load(ROOT/'assets/server_rack/server-rack.glb')
parts=[]
for idx,node in enumerate(sorted(scene.graph.nodes_geometry)):
    transform,gname=scene.graph[node]
    mesh=scene.geometry[gname].copy(); mesh.apply_transform(transform)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[1,0,0]))
    if node=='server-rack_body':
        # Keep the imported cabinet shell; replace its fused static equipment with
        # separate full-depth articulated servers. No duplicate fronts remain.
        c=mesh.triangles_center
        keep=(np.abs(c[:,0])>.265)|(c[:,2]<.105)|(c[:,2]>1.91)|(c[:,1]>.48)
        mesh.update_faces(keep)
        mesh.remove_unreferenced_vertices()
    # Explicit OBJ UVs/normals; materials are bound in MJCF below.
    obj,_=trimesh.exchange.obj.export_obj(mesh,include_texture=True,include_normals=True,return_texture=True)
    (OUT/f'rack_{idx}.obj').write_text(obj)
    mn=f'rack_mesh_{idx}'; mat=f'rack_mat_{idx}'
    add(assets,'mesh',name=mn,file=f'assets/converted/rack_{idx}.obj')
    pbr=mesh.visual.material
    attrs={}
    if pbr.baseColorTexture is not None:
        a=np.array(pbr.baseColorTexture.convert('RGB')).astype(float)/255
        # Refinish the neutral cream source cabinet in charcoal; retain coloured accents.
        neutral=(a.max(axis=2)-a.min(axis=2))<.14
        a[neutral]*=np.array([.23,.26,.30])
        Image.fromarray((a*255).astype('uint8')).save(OUT/f'rack_{idx}.png')
        add(assets,'texture',name=f'rack_tex_{idx}',type='2d',file=f'assets/converted/rack_{idx}.png')
        attrs['texture']=f'rack_tex_{idx}'
    else:
        rgba=np.asarray(pbr.baseColorFactor if pbr.baseColorFactor is not None else [255]*4)/255
        attrs['rgba']=' '.join(map(str,rgba))
    add(assets,'material',name=mat,**attrs,specular='.3',shininess='.35',emission='.55' if 'led' in node else '0')
    parts.append((node,mn,mat))

hardware_materials=hardware_assets(assets,OUT)

world=add(root,'worldbody')
for i,(pos,direction) in enumerate([('0 -3 6','0 .3 -1'),('3 1 5','-.4 -.3 -1'),('-3 -1 4','.4 0 -1')]):
    add(world,'light',name=f'studio_{i}',pos=pos,dir=direction,diffuse='.35 .37 .4',castshadow='true' if i==0 else 'false',directional='true')
add(world,'geom',name='ground',type='plane',pos='0 0 -.03',size='8 8 .1',rgba='.4 .45 .5 1')
for ix in range(-5,6):
    for iy in range(-5,4):
        box(world,f'tile_{ix}_{iy}',f'{ix*.6} {iy*.6} -.014','.298 .298 .014','floor_a' if (ix+iy)%2 else 'floor_b',contype='0',conaffinity='0')
for x in [-1.62,1.62]:
    box(world,f'aisle_mark_{x}',f'{x} -1.25 .002','.022 1.3 .002','line',contype='0',conaffinity='0')
for rack_i,x in enumerate([-.96,-.32,.32,.96]):
    rack=add(world,'body',name=f'rack_{rack_i+1:02}',pos=f'{x} 0 0')
    for node,mn,mat in parts:
        # Front doors removed for service; original static LEDs belong to old units.
        if 'door' in node or 'led' in node: continue
        add(rack,'geom',type='mesh',mesh=mn,material=mat,contype='0',conaffinity='0',group='2')
    # Separate thin frame/back/rail colliders; never use a whole-rack convex hull.
    for side in [-1,1]:
        box(rack,f'r{rack_i}_side_{side}',f'{side*.285} .02 1','.015 .49 .95','graphite',group='3')
    box(rack,f'r{rack_i}_back','0 .51 1','.27 .015 .96','graphite',group='3')
    box(rack,f'r{rack_i}_top','0 .02 1.98','.27 .49 .015','graphite',group='3')
    populate(rack,rack_i,hardware_materials)
    name=f'rack_label_{rack_i}'
    label(f'R{rack_i+1:02} / A',name)
    add(assets,'texture',name=name,type='2d',file=f'assets/converted/{name}.png')
    add(assets,'material',name=name,texture=name,texuniform='false')
    box(rack,name,'0 -.476 1.935','.245 .004 .055',name,contype='0',conaffinity='0')

# Service base: a carriage on a floor rail along the aisle (left/right) carrying a lift
# (up/down) with the Panda on top. Both are position-servoed slide joints; robot/arm/base.py
# computes their targets from rack and shelf positions. The rail sits far enough back that
# the tucked arm clears the rack fronts (y = -.47) while it travels.
for y in [RAIL_Y-.12,RAIL_Y+.12]:
    box(world,f'floor_rail_{y:.2f}',f'0 {y} .006','1.75 .018 .006','metal',contype='0',conaffinity='0')
cart=add(world,'body',name='service_cart',pos=f'0 {RAIL_Y} 0')
add(cart,'joint',name='rail_x',type='slide',axis='1 0 0',range=f'{RAIL_RANGE[0]} {RAIL_RANGE[1]}',damping='150')
# Footprint runs from 0.12 m on the tray side to 0.34 m on the mast side of the arm base
# (narrower on the tray side so the hand clears it when pulling the lowest trays).
box(cart,'cart_body','.11 0 .105','.23 .2 .075','yellow')
box(cart,'cart_status','.11 -.202 .12','.16 .004 .02','cyan',contype='0',conaffinity='0')
box(cart,'lift_mast','.30 0 1.125','.035 .07 .995','graphite')
lift=add(cart,'body',name='service_lift',pos=f'0 0 {LIFT_LOW}')
add(lift,'joint',name='lift_z',type='slide',axis='0 0 1',range=f'0 {LIFT_TRAVEL}',damping='150')
box(lift,'lift_platform','.12 0 -.012','.15 .13 .012','metal')
robot_mount=add(lift,'body',name='robot_mount',pos='0 0 0',euler='0 0 1.57079632679')
robot=ET.parse(ROOT/'assets/franka_emika_panda/panda.xml').getroot()
for child in robot:
    child=copy.deepcopy(child)
    if child.tag in ['compiler','option','keyframe']: continue
    if child.tag=='asset':
        for item in child:
            if 'file' in item.attrib:
                # Preserve implicit mesh names before prefixing their file paths.
                if item.tag=='mesh' and 'name' not in item.attrib:item.set('name',Path(item.get('file')).stem)
                item.set('file','assets/franka_emika_panda/assets/'+item.get('file'))
            assets.append(item)
    elif child.tag=='worldbody':
        for item in child: robot_mount.append(item)
    else: root.append(child)
# Rail and lift servos. The lift's constant bias cancels the weight it carries
# (filled in after a first compile below), so it holds height without sagging.
act=add(root,'actuator')
add(act,'general',name='rail_x',joint='rail_x',biastype='affine',gainprm=BASE_KP,biasprm=f'0 -{BASE_KP} -{RAIL_KV}',ctrlrange=f'{RAIL_RANGE[0]} {RAIL_RANGE[1]}',forcerange='-800 800')
lift_act=add(act,'general',name='lift_z',joint='lift_z',biastype='affine',gainprm=BASE_KP,biasprm=f'0 -{BASE_KP} -{LIFT_KV}',ctrlrange=f'0 {LIFT_TRAVEL}')
# Home: base parked beside the right-hand rack, arm folded in its travel pose.
TUCK='0 -1.0 0 -3.0 0 2.1 .7853'
# The stock Panda gripper squeezes ~1 N on a thin handle: too weak to drag a tray.
for a in root.iter('general'):
    if a.get('name')=='actuator8':
        a.set('gainprm',f'{GRIP_KP*(.04+GRIP_PAST)/255:.6f} 0 0');a.set('biasprm',f'{-GRIP_KP*GRIP_PAST:g} -{GRIP_KP} -{GRIP_KP//100}')
keys=add(root,'keyframe')
add(keys,'key',name='home',qpos=('0 '*(4*COUNT))+f'{HOME_X} 0 {TUCK} .04 .04',ctrl=f'{TUCK} 255 {HOME_X} 0')
# Camera axes computed from a look-at target, in world coordinates.
def camera(name,pos,target,fovy):
    z=np.array(pos)-target; z=z/np.linalg.norm(z)
    x=np.cross([0,0,1],z); x=x/np.linalg.norm(x); y=np.cross(z,x)
    add(world,'camera',name=name,pos=' '.join(map(str,pos)),xyaxes=' '.join(map(str,np.r_[x,y])),fovy=fovy)
camera('overview',[3.45,-5.1,3.1],np.array([0,-.25,.95]),'40')
camera('front',[0,-4.8,1.9],np.array([0,0,1.0]),'39')
camera('service',[-1.55,-2.5,1.9],np.array([-.32,-.25,1.0]),'47')
camera('hardware',[2.0,-2.4,2.15],np.array([.96,-.75,1.0]),'42')
ET.indent(root)
ET.ElementTree(root).write(ROOT/'datacentre.xml',encoding='unicode')
import mujoco
m=mujoco.MjModel.from_xml_path(str(ROOT/'datacentre.xml'))
weight=float(m.body_subtreemass[m.body('service_lift').id])*-float(m.opt.gravity[2])
lift_act.set('biasprm',f'{weight:.3f} -{BASE_KP} -{LIFT_KV}')
lift_act.set('forcerange',f'{weight-800:.1f} {weight+800:.1f}')
ET.ElementTree(root).write(ROOT/'datacentre.xml',encoding='unicode')
print('Built',ROOT/'datacentre.xml',f'(lift carries {weight:.1f} N)')
