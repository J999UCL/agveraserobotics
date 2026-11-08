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

def label(text, name):
    im=Image.new('RGB',(512,128),'#15212e'); d=ImageDraw.Draw(im)
    font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',48)
    d.rectangle((0,0,12,128),fill='#46cfdd')
    d.text((36,34),text,font=font,fill='#e4f4f6')
    im.save(OUT / f'{name}.png')

root=ET.Element('mujoco',model='Agverase data centre')
add(root,'compiler',angle='radian',autolimits='true')
add(root,'option',timestep='0.002',integrator='implicitfast')
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

# Fixed service cart: a mounting platform, not a mobile-base controller.
cart=add(world,'body',name='service_cart',pos='-.32 -1.27 0')
box(cart,'cart_body','0 0 .245','.32 .38 .135','yellow')
box(cart,'cart_top','0 0 .393','.34 .4 .018','graphite')
box(cart,'cart_bumper','0 0 .105','.34 .4 .025','rubber')
for x in [-.265,.265]:
    for y in [-.27,.27]:
        add(cart,'geom',name=f'wheel_{x}_{y}',type='cylinder',pos=f'{x} {y} .09',size='.085 .035',euler='0 1.5708 0',material='rubber')
for x in [-.29,.29]:
    box(cart,f'handle_post_{x}',f'{x} -.34 .59','.012 .012 .18','metal')
box(cart,'handle_bar','0 -.34 .77','.3 .014 .014','metal')
box(cart,'cart_status','0 -.382 .25','.2 .004 .025','cyan',contype='0',conaffinity='0')
robot_mount=add(cart,'body',name='robot_mount',pos='0 .06 .412',euler='0 0 1.57079632679')
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
keys=add(root,'keyframe')
add(keys,'key',name='home',qpos=('0 '*(4*COUNT))+'0 0 0 -1.57079 0 1.57079 -.7853 .04 .04',ctrl='0 0 0 -1.57079 0 1.57079 -.7853 255')
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
print('Built',ROOT/'datacentre.xml')
