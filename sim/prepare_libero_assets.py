"""Scene preparation only: compact visuals and registered LIBERO fixture XML."""
from pathlib import Path
import copy,json,os,xml.etree.ElementTree as ET
import numpy as np
import trimesh
from PIL import Image
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'tasks'/'assets'

def build():
 OUT.mkdir(parents=True,exist_ok=True)
 tree=ET.parse(ROOT/'datacentre.xml').getroot();assets=tree.find('asset')
 materials={x.get('name'):x for x in assets.findall('material')};meshes={x.get('name'):x for x in assets.findall('mesh')}
 sample=tree.find('.//body[@name="r01_u01"]');parts=[];colors=[]
 for g in sample.findall('geom'):
  if g.get('group')=='3' or '_handle' in g.get('name',''):continue
  if g.get('type')=='mesh':
   m=trimesh.load(ROOT/meshes[g.get('mesh')].get('file'),force='mesh',process=False)
  elif g.get('type')=='box':
   m=trimesh.creation.box(extents=np.array(list(map(float,g.get('size').split())))*2);m.apply_translation(list(map(float,g.get('pos','0 0 0').split())))
  else:continue
  mat=materials[g.get('material')];rgba=list(map(float,mat.get('rgba','.5 .5 .5 1').split()))
  colors.append([int(c*255) for c in rgba[:3]])
  m.visual=trimesh.visual.TextureVisuals(uv=np.tile([(len(colors)-.5)/32,.5],(len(m.vertices),1)))
  parts.append(m)
 palette=np.zeros((8,256,3),dtype=np.uint8)
 for i,color in enumerate(colors):palette[:,i*8:(i+1)*8]=color
 Image.fromarray(palette).save(OUT/'server-palette.png')
 mesh=trimesh.util.concatenate(parts);obj,_=trimesh.exchange.obj.export_obj(mesh,include_texture=True,include_normals=True,return_texture=True);(OUT/'server.obj').write_text(obj.rstrip()+'\n')
 for i,x in enumerate([-.64,0.,.64,1.28]):
  src=copy.deepcopy(tree.find('.//body[@name="rack_%02d"]'%(i+1)))
  src.set('name','object');src.set('pos','%s -.8 0'%x);src.set('quat','0 0 0 1')
  root=ET.Element('mujoco');aa=ET.SubElement(root,'asset')
  # Original cabinet meshes and materials remain available; no robot assets.
  for a in assets:
   n=a.get('name','')
   if n.startswith(('rack','server_','hw_')) or n in ('metal','graphite','white_trim','cyan'):
    a=copy.deepcopy(a)
    if a.get('file'):a.set('file',os.path.relpath(ROOT/a.get('file'),OUT))
    aa.append(a)
  ET.SubElement(aa,'mesh',name='compact_server',file='server.obj')
  ET.SubElement(aa,'texture',name='compact_tex',type='2d',file='server-palette.png')
  ET.SubElement(aa,'material',name='compact_material',texture='compact_tex')
  for body in src.findall('body'):
   name=body.get('name','')
   if '_u' not in name:continue
   for g in list(body.findall('geom')):
    # Retain the real collision shells and handles; handles carry the target cue.
    if g.get('group')!='3' and '_handle' not in g.get('name',''):body.remove(g);continue
    g.set('group','0' if g.get('group')=='3' else '1')
    if g.get('contype','1')!='0':g.set('contype','1');g.set('conaffinity','1')
   ET.SubElement(body,'geom',name=name+'_visual',type='mesh',mesh='compact_server',material='compact_material',group='1',contype='0',conaffinity='0',mass='0')
   slot=int(name.split('_u')[1]);ET.SubElement(body,'site',name='u%02d_region'%slot,quat='1 0 0 0',pos='0 0 .04',size='.23 .37 .04',type='box',rgba='0 0 0 0')
   if i==1 and slot==21:
    for g in body.findall('geom'):
     if g.get('name','').endswith('_handle_1'):g.set('material','cyan')
  for g in src.findall('geom'):
   g.set('group','0' if g.get('group')=='3' else '1')
  world=ET.SubElement(root,'worldbody');outer=ET.SubElement(world,'body');outer.append(src)
  for name,pos,size in [('bottom_site','0 0 0','0.001'),('top_site','0 0 2','0.001'),('horizontal_radius_site','.3 .53 0','0.001')]:ET.SubElement(outer,'site',name=name,pos=pos,size=size,rgba='0 0 0 0')
  ET.ElementTree(root).write(OUT/('rack_%s.xml'%chr(97+i)),encoding='unicode')
 bddl='''(define (problem LIBERO_Tabletop_Manipulation)
 (:domain robosuite)
 (:language Pull out the server with the cyan handle)
 (:regions (u21_region (:target data_center_rack_b_1)))
 (:fixtures main_table - table
 data_center_rack_a_1 - data_center_rack_a
 data_center_rack_b_1 - data_center_rack_b
 data_center_rack_c_1 - data_center_rack_c
 data_center_rack_d_1 - data_center_rack_d)
 (:objects)
 (:obj_of_interest data_center_rack_b_1_u21_region)
 (:init (Close data_center_rack_b_1_u21_region))
 (:goal (And (Open data_center_rack_b_1_u21_region))))
'''
 (ROOT/'tasks'/'pull_server.bddl').write_text(bddl)
 print('Prepared LIBERO rack fixtures:',OUT)
if __name__=='__main__':build()
