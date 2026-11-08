# Asset provenance

## Server rack — Grab3D

- Source: https://grab3d.com/models/electronics/server-rack/
- Download: https://grab3d.com/dl/server-rack/server-rack.glb
- Licence: **CC0-1.0**; upstream notice is bundled at `assets/server_rack/LICENSE.txt` and metadata at `assets/server_rack/manifest.json`.
- Source dimensions: 0.600 × 1.065 × 2.001 metres; 27,252 triangles.
- Source disclosure: procedurally built using Blender scripts with AI assistance in writing code; no generated geometry/textures (see upstream licence notice).
- Local adaptations: Y-up to Z-up conversion, OBJ/UV export, neutral cabinet texture recoloured charcoal, repetition into four racks, central two doors omitted. Source GLB remains intact.
- Separate LEDs and glass materials retained; MuJoCo uses base colour, transparency and approximate specular/emission, rather than the complete glTF PBR shader.

## Franka Emika Panda — MuJoCo Menagerie

- Source: https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda
- Upstream revision at download: `98a70c569c4c1b6da467a20012f766ee1f8dc521`.
- Licence: **Apache-2.0**; licence, README and changelog are bundled in `assets/franka_emika_panda/`.
- The supplied model, visual meshes, collision meshes, gripper, actuator definitions and home pose are reused. Scene composition changes the root mounting transform and prefixes mesh paths; upstream source files remain intact.

## Locally composed elements

Cart body/handle/wheels, floor tiles, safety lines, rack labels, camera placement, lighting and simple rack collision proxies were composed in `build_scene.py`. The cart is a visual fixed mounting platform with no locomotion model.

`assets/checksums.json` records SHA-256 checksums for the downloaded source files. Converted meshes/textures and the scene can be regenerated with `./sim/setup.sh`.

## Articulated server update

The imported rack's fused internal equipment and LEDs are removed from the composed scene; only its cabinet shell remains. Front doors are omitted on all four service racks. The original GLB is preserved intact.

`server_hardware.py` composes new simplified 2U server hardware from shared box, cylinder and annulus meshes: chassis, eight drive carriers/drives, fan wall, PCB, CPU heatsinks, DIMMs, power supplies and connectors. These hardware meshes are locally authored, not downloaded vendor CAD. All 80 server assemblies share the mesh assets, with independent rail joints and separate simple collision geometry. The source model checksums remain applicable to the unmodified downloaded files.
