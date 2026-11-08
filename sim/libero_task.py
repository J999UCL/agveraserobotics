"""Documented LIBERO object registration. No robot, controller or env overrides."""
import os
from pathlib import Path
from robosuite.models.objects import MujocoXMLObject
from libero.libero.envs.base_object import register_object
ROOT=Path(os.environ.get('AGVERASE_RACK_ASSETS',str(Path(__file__).resolve().parent/'tasks'/'assets')))

class RackAsset(MujocoXMLObject):
    asset_letter=None
    def __init__(self,name,obj_name=None,joints=None):
        super().__init__(str(ROOT/('rack_'+self.asset_letter+'.xml')),name=name,joints=None,obj_type='all',duplicate_collision_geoms=False)
        self.category_name='data_center_rack_'+self.asset_letter
        self.rotation=(0,0);self.rotation_axis='z'
        self.object_properties={'articulation':{'default_open_ranges':[.6,.72],'default_close_ranges':[0,.001]},'vis_site_names':{}}
    def is_open(self,qpos):return qpos>.6
    def is_close(self,qpos):return qpos<.005

@register_object
class DataCenterRackA(RackAsset):asset_letter='a'
@register_object
class DataCenterRackB(RackAsset):asset_letter='b'
@register_object
class DataCenterRackC(RackAsset):asset_letter='c'
@register_object
class DataCenterRackD(RackAsset):asset_letter='d'
