import ast,json,tempfile,unittest
from pathlib import Path
import xml.etree.ElementTree as ET
from episode_catalog import episode_catalog
ROOT=Path(__file__).resolve().parent
class ContractTests(unittest.TestCase):
 def test_no_project_motor_controller(self):
  self.assertFalse((ROOT/'agent.py').exists());self.assertFalse((ROOT/'native_env.py').exists())
  client=(ROOT/'libero_client.py').read_text();ast.parse(client)
  for forbidden in ['qfrc_applied','mj_step','PolicyFrame','set_joint_qpos','data.ctrl','set_goal','class Controller']:
   self.assertNotIn(forbidden,client)
  cloud=(ROOT/'cloud.py').read_text();self.assertNotIn('from agent import',cloud);self.assertNotIn(".read_text().replace('for task",cloud)
 def test_fixture_joint_and_site_contract(self):
  count=0
  for p in (ROOT/'tasks'/'assets').glob('rack_*.xml'):
   root=ET.parse(p).getroot();count+=len(root.findall('.//joint'))
   self.assertEqual(len(root.findall(".//site[@name='u21_region']")),1)
   self.assertIsNotNone(root.find(".//site[@name='u21_region']").get('quat'))
   for asset in root.findall('.//*[@file]'):self.assertTrue((p.parent/asset.get('file')).exists())
   for geom in root.findall('.//geom'):self.assertIn(geom.get('group'),['0','1'])
  self.assertEqual(count,80)
 def test_catalog_ignores_non_recording_json(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'manifest.json';p.write_text('');self.assertEqual(episode_catalog(d),[])
   p.write_text('{"slug":"rack"}');self.assertEqual(episode_catalog(d),[])
if __name__=='__main__':unittest.main()
