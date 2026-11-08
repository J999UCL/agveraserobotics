"""Repeated 19-inch 2U server assemblies: shared detailed meshes, separate physics."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import trimesh

COUNT=20
PITCH=.0889
LOW=.12
TRAVEL=.72

def el(p,tag,**kw):return ET.SubElement(p,tag,{k:str(v) for k,v in kw.items()})
def box(p,name,pos,size,mat,**kw):return el(p,'geom',name=name,type='box',pos=pos,size=size,material=mat,**kw)

def hardware_assets(assets,out):
    colors={'pcb':'.025 .22 .12 1','solder':'.63 .66 .59 1','chip':'.025 .029 .032 1','alu':'.58 .63 .68 1','copper':'.65 .35 .12 1','socket':'.13 .15 .18 1','ram':'.035 .28 .16 1','blue':'.035 .35 .7 1','led':'.12 .85 .43 1','chassis':'.27 .31 .35 1'}
    for name,rgba in colors.items():el(assets,'material',name='hw_'+name,rgba=rgba,specular='.35',shininess='.4',emission='.4' if name=='led' else '0')
    groups={k:[] for k in colors}
    def b(mat,c,s):
        m=trimesh.creation.box(extents=s);m.apply_translation(c);groups[mat].append(m)
    def cyl(mat,c,r,h,sections=12):
        m=trimesh.creation.cylinder(radius=r,height=h,sections=sections);m.apply_translation(c);groups[mat].append(m)
    # Open top chassis, folded side panels, rear IO panel and perforated front.
    b('chassis',[0,0,.003],[.438,.76,.006])
    for x in [-.218,.218]:b('chassis',[x,0,.04],[.004,.76,.074])
    b('chassis',[0,.377,.038],[.436,.006,.07])
    b('chip',[0,-.382,.04],[.48,.012,.078])
    # Eight front hot-swap drive carriers; vent gaps are geometry, not a painted plane.
    for ix in range(4):
        for iz in range(2):
            x=-.164+ix*.109;z=.022+iz*.035
            b('socket',[x,-.391,z],[.1,.008,.03])
            b('alu',[x,-.396,z],[.085,.004,.022])
            b('blue',[x+.039,-.401,z],[.006,.007,.022])
            for v in range(6):b('chip',[x-.033+v*.011,-.399,z],[.006,.003,.014])
            b('led',[x-.043,-.403,z+.008],[.003,.003,.004])
            b('alu',[x,-.265,.025+iz*.028],[.098,.22,.024])
            b('chip',[x,-.261,.038+iz*.028],[.075,.16,.002])
    # Motherboard and two CPU sockets / finned heatsinks.
    b('pcb',[0,.10,.012],[.4,.26,.009])
    for x in [-.07,.07]:
        b('socket',[x,.125,.021],[.074,.088,.012])
        b('copper',[x,.125,.03],[.064,.077,.007])
        for fin in range(16):b('alu',[x-.027+fin*.0036,.125,.049],[.0015,.075,.034])
    # Eight DIMMs in separate slots, with chips and retaining clips.
    for x in [-.193,-.177,-.161,-.145,.145,.161,.177,.193]:
        b('socket',[x,.125,.02],[.009,.13,.009])
        b('ram',[x,.125,.04],[.003,.124,.034])
        for j in range(8):b('chip',[x-.003,.074+j*.014,.042],[.003,.01,.019])
        for y in [.058,.192]:b('alu',[x,y,.03],[.008,.005,.018])
    # Central fan wall, shrouds and visible radial blades (horizontal axes).
    for x in [-.175,-.105,-.035,.035,.105,.175]:
        b('socket',[x,-.085,.041],[.065,.065,.067])
        fan=trimesh.creation.annulus(r_min=.025,r_max=.03,height=.006,sections=20)
        fan.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[1,0,0]));fan.apply_translation([x,-.121,.041]);groups['alu'].append(fan)
        for a in np.linspace(0,2*np.pi,7,endpoint=False):
            m=trimesh.creation.box(extents=[.018,.003,.007]);m.apply_transform(trimesh.transformations.rotation_matrix(a,[0,1,0]));m.apply_translation([x+.014*np.cos(a),-.125,.041+.014*np.sin(a)]);groups['chip'].append(m)
        m=trimesh.creation.cylinder(radius=.007,height=.008,sections=12);m.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[1,0,0]));m.apply_translation([x,-.126,.041]);groups['alu'].append(m)
        b('blue',[x,-.085,.078],[.025,.015,.005])
    # Rear redundant PSUs and connectors.
    for x in [-.15,-.045]:
        b('alu',[x,.31,.048],[.093,.13,.049])
        b('chip',[x,.382,.05],[.055,.005,.025])
        for j in range(6):b('chip',[x-.034+j*.013,.315,.074],[.004,.09,.001])
        b('blue',[x+.034,.35,.076],[.012,.032,.006])
    for x in [.04,.075,.11,.145]:b('solder',[x,.382,.027],[.025,.012,.018])
    # Board traces, capacitors, PCIe slots and support standoffs.
    for i in range(12):
        b('solder',[-.19+i*.029,.055,.017],[.001,.03,.001])
        cyl('solder',[-.18+i*.029,.01,.023],.003,.014)
    for x in [.06,.09]:b('socket',[x,.21,.023],[.009,.03,.014])
    for x in [-.19,.18]:
        for y in [-.02,.22]:cyl('alu',[x,y,.013],.004,.018)
    for name,meshes in groups.items():
        mesh=trimesh.util.concatenate(meshes);mesh.export(out/f'server_{name}.obj')
        el(assets,'mesh',name='server_'+name,file=f'assets/converted/server_{name}.obj')
    return list(groups)

def populate(rack,rack_i,materials):
    prefix=f'r{rack_i+1:02}'
    # Four 19-inch mounting uprights.
    for x in [-.25,.25]:
        for y in [-.405,.39]:
            box(rack,f'{prefix}_post_{x}_{y}',f'{x} {y} 1.025','.009 .012 .91','metal')
    for n in range(COUNT):
        z=LOW+n*PITCH
        ident=f'{prefix}_u{n*2+1:02}'
        # Stationary outer rail pair, kept clear of the sliding chassis.
        for side in [-1,1]:
            box(rack,f'{ident}_fixedrail_{side}',f'{side*.235} 0 {z+.01}','.005 .37 .007','metal',contype='0',conaffinity='0')
        tray=el(rack,'body',name=ident,pos=f'0 -.025 {z}')
        el(tray,'joint',name=ident+'_slide',type='slide',axis='0 -1 0',range=f'0 {TRAVEL}',damping='18',frictionloss='4',armature='.1',limited='true',solreflimit='.015 1')
        el(tray,'inertial',pos='0 0 .035',mass='12',diaginertia='.58 .20 .78')
        for mat in materials:el(tray,'geom',type='mesh',mesh='server_'+mat,material='hw_'+mat,contype='0',conaffinity='0',group='2')
        # Simple shell colliders and graspable handles. Trays collide with the robot/rack,
        # but adjacent fixed rail levels do not waste time on tray-to-tray checks.
        for name,pos,size in [('floor','0 0 .003','.217 .377 .003'),('left','-.218 0 .04','.002 .377 .035'),('right','.218 0 .04','.002 .377 .035'),('front','0 -.382 .04','.24 .006 .039')]:
            box(tray,ident+'_'+name,pos,size,'hw_chassis',group='3',contype='2',conaffinity='1')
        for side in [-1,1]:
            x=side*.225
            for dz in [.014,.066]:box(tray,f'{ident}_handlepost_{side}_{dz}',f'{x} -.418 {dz}','.007 .027 .006','metal',contype='2',conaffinity='1')
            box(tray,f'{ident}_handle_{side}',f'{x} -.444 .04','.007 .007 .032','white_trim',contype='2',conaffinity='1')
            box(tray,f'{ident}_innerrail_{side}',f'{side*.227} 0 .018','.004 .36 .008','metal',contype='0',conaffinity='0')
