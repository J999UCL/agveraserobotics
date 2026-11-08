# MuJoCo direction: robotics hackathon discussion

## Implementation update — 2026-11-08

The implemented execution split is now full simulation and pretrained OpenPI inference on Modal `j999ucl`, with saved-state 3D playback on the Mac. The active path uses the supplied LIBERO MountedPanda, controller, gripper, cameras and reset through vendor APIs. The custom Menagerie controller and coordinate bridge have been removed. Four registered rack fixtures contain 80 sliding servers. Standard OpenPI policy recording and robosuite `DataCollectionWrapper` retain inputs, outputs, states, actions, model XML and portable assets. Replay uses the recorded MuJoCo 3.2.3 version without advancing physics.

Measured: the unchanged `libero_goal` example succeeded on **10/10 tasks**; the new rack task completed **0/3 extractions** (later state audit: zero actual tray displacement; the earlier 0.226 mm was a reset offset). The native controller passes hold and axis checks, but downward/away motion remains in policy rollouts. Task/visual distribution mismatch is plausible, not conclusively isolated. This does not demonstrate server maintenance. The original $5 compute allowance remains in force; jobs are stopped. See `sim/README.md` for commands and `sim/RACK_DIAGNOSIS.md` for the controlled follow-up: familiar wording and less visual clutter improve approach, but ten short runs, including a closer target pose, still make no target-tray contact. Gemini orchestration has not been added by this work.

Checkpoint correction to the historical description below: pinned `pi05_libero` supplies numeric state in the API, but its `discrete_state_input=False` configuration does not condition this π0.5 network on that numeric state. Its conditioning is images and language.

The remainder is the earlier design discussion and historical status.

## Objective and current direction

Four-person, 24-hour robotics hackathon; Modal GPU compute available. We want a commercially plausible embodied agent that perceives the environment, makes decisions through an LLM, physically acts, and adapts to an unexpected change. The current direction is **MuJoCo locally on the Mac, a real VLA for manipulation, and Gemini Robotics-ER above it for task reasoning**. This is a proposed architecture, not an implemented or validated system. The data-centre application remains under discussion.

The rubric weights perception-driven action 25%, adaptation 25%, operational autonomy 15%, relevance/impact 20%, and ambition/technical quality 15%. Show sensor evidence alongside actions and a mid-sequence change in one continuous run. Deliverables: one-page technical PDF, up to one-minute continuous simulation video, and five-minute pitch. Remote model inference trades off against the rubric's preference for local/edge decisions; document where models run and measured latency.

## Application: remote data-centre maintenance

The motivation is reducing human visits to remote facilities. Inspection/reporting alone is insufficient for the intended demo: the robot should physically resolve a maintenance task. Candidate work includes replacing a drive, fan module or removable cooling cartridge, with identification, diagnosis, spare selection, execution and verification handled as one incident. A useful agent-level exception is finding an incompatible spare or an asset label that disagrees with the work order.

We rejected contrived faults such as a trolley trapping a cooling hose as too niche and awkward to diagnose. Complex cooling, power and leak incidents are real application areas, but do not themselves establish a feasible robotic repair skill. The immediate physical prototype discussed was an accessible, unlatched module with a handle, extracted along a guide and exchanged with a spare. This is a simplified interaction, not yet a demonstrated VLA capability.

**Main unresolved issue:** data-centre manipulation is likely outside the ready-made policy's training distribution. Changing simulator does not teach latch release, constrained extraction, connector insertion or RAM replacement. For a hackathon we should reproduce an existing learned skill first, then adapt objects and scene incrementally. Preserve a working baseline rather than betting the entire demo on unseen fine manipulation.

## What the models actually provide

### Gemini Robotics-ER: agent and visual reasoning

The repository's `.env` contains a populated `gemini_api_key`. A read-only Google model-list request authenticated successfully and listed `gemini-robotics-er-2-preview` and `gemini-robotics-er-2-streaming-preview`. **No inference request was made**, so quota and successful inference remain unverified. Never print or commit the key.

Robotics-ER interprets images, reasons spatially, plans and calls tools. It does not supply a general motor-control policy. Google's actual Gemini Robotics 2 VLA is in private preview via the [Trusted Tester waitlist](https://docs.google.com/forms/d/1sM5GqcVMWv-KmKY3TOMpVtQ-lDFeAftQ-d9xQn92jCE/viewform); it was not listed for our key. No application was submitted. Sources: [ER](https://deepmind.google/models/gemini-robotics/embodied-reasoning/), [VLA](https://deepmind.google/models/gemini-robotics/vla/).

### Physical Intelligence π0.5: preferred first manipulation experiment

π0.5 is a vision-language-action model: images, language and robot state condition a short sequence of continuous robot actions. A vision-language backbone supplies scene/instruction representations and a flow-matching action head generates actions. The controller executes a portion, observes again and repeats. It is not merely selecting a handwritten `pick()` routine.

[OpenPI](https://github.com/Physical-Intelligence/openpi) provides public base and task/platform-specific checkpoints. The concrete starting point is **`pi05_libero`**, already fine-tuned for LIBERO, with a supplied MuJoCo/robosuite client. The client supplies an external camera, wrist camera, end-effector/gripper state and instruction; predicted actions are passed into the simulation. A WebSocket policy server supports separating the model environment from the simulator.

The authors report 96.85% average success across four LIBERO suites for their supplied fine-tuned checkpoint. **This measures those benchmark tasks, not our custom maintenance task.** Base `pi05_base`, LIBERO `pi05_libero`, and DROID `pi05_droid` are distinct choices; do not assume a fine-tuned benchmark policy inherits unrestricted capability from the research demos. OpenPI lists >8 GB VRAM for inference, >22.5 GB for LoRA fine-tuning and >70 GB for full fine-tuning as general estimates; actual checkpoint/runtime use must be measured. Its supported/tested platform is Ubuntu 22.04, so plan Linux GPU inference rather than native Mac inference. Sources: [LIBERO recipe/results](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md), [client](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py), [research](https://www.pi.website/blog/pi05).

π0.5's research demonstrates transferring skills to unfamiliar settings; that is different from learning entirely new contact-intensive repair skills. Our initial aim is to reproduce the supplied task **without training**, then test transfer. Training a new maintenance skill is a separate workstream, not assumed necessary or guaranteed feasible within the hackathon.

### NVIDIA GR00T: alternative, not the selected baseline

[GR00T N1.7](https://github.com/NVIDIA/Isaac-GR00T) also predicts actions from vision, language and robot state. Public base and benchmark-specific weights exist; the repository says its Cosmos-Reason2-2B backbone requires Hugging Face access. NVIDIA lists 16 GB+ VRAM for inference, excluding simulation requirements. Its FAQ suggests roughly 100–500 demonstration episodes for new fine-manipulation tasks, plus iterative correction data for difficult tasks. These are guidelines, not a mandatory amount for reproducing a supplied checkpoint.

NVIDIA supplies a specific G1 apple-to-plate Isaac simulation using bundled policies. It is a concrete baseline, not proof of general maintenance. We preferred investigating π0.5 + LIBERO because model and MuJoCo environment are already paired. No GR00T weights were downloaded or policies run. Sources: [FAQ](https://github.com/NVIDIA/Isaac-GR00T/blob/main/FAQ.md), [Isaac example](https://docs.nvidia.com/isaac-os/tutorials/gr00t-wbc).

## Why the Gemini/Isaac example is not general learned grasping

We inspected [ksminmin/IsaacSim_GeminiRobotics](https://github.com/ksminmin/IsaacSim_GeminiRobotics). Gemini selects tools such as pick/place; a handwritten FR3 inverse-kinematics state machine performs approach, descend, gripper closure, lift and placement. Precise object poses come from simulator-published transforms. It supports the supplied blocks and robot configuration, not arbitrary maintenance objects.

The controller commands finger positions of 0.015 and later treats a measured position above 0.01 as grasp success, which could produce false positives. That check needs validation/replacement before relying on it. The README's “VLA brain” wording should not be interpreted as Gemini producing learned motor actions. Sources: [controller](https://github.com/ksminmin/IsaacSim_GeminiRobotics/blob/main/wsl_ws/src/isaac_ros2_control/isaac_ros2_control/multi_robot_controller.py), [scene/TF publishing](https://github.com/ksminmin/IsaacSim_GeminiRobotics/blob/main/isaacsim_scripts/three_robot_tower.py).

## Environment and premade assets

**User preference: reuse premade assets. Do not model an entire data centre from scratch or make Blender a prerequisite.** Download existing rack/server/room meshes, arrange them in MuJoCo, and reuse the supplied robot, gripper and cameras. One candidate found is [Spellkaze's downloadable server rack](https://sketchfab.com/3d-models/server-rack-62f6779cb7e448b19aaf58544c3c7218), marked CC Attribution. No asset was downloaded or vetted in simulation; retain attribution and check each asset's licence and actual geometry.

The environment runs in MuJoCo. Blender would only be an optional conversion/editing tool, not the simulator, and is unnecessary if assets can be imported directly. Existing visual assets still need MuJoCo configuration: scale, mass, collision geometry, joints and actuators. Moving drive sleds and doors must be separate bodies. Visual detail can come from meshes while simple separate collision shapes define contact. MuJoCo's convex mesh collision hull can fill rack openings, so avoid one solid collision hull around an entire cabinet. [Mesh documentation](https://mujoco.readthedocs.io/en/3.6.0/XMLreference.html).

Start with one interactive rack/work area within the existing arm's reach, a few labelled modules and a spare tray; add surrounding racks for context. Building the scene is separate from proving the policy can manipulate it. Do not claim the scene or imported assets are already functional.

## Mac setup and proposed execution split

Read-only inspection found an Apple Silicon ARM64 Mac, `uv` installed, and no MuJoCo, robosuite or PyTorch in the inspected project virtual environment. System Python reported 3.14.7; use an isolated Python version compatible with the selected LIBERO dependency stack rather than assuming the system interpreter is suitable.

**MuJoCo itself supports native macOS ARM64**, including local interactive rendering; no NVIDIA GPU, Docker, Modal or remote desktop is needed just to run the simulator. The official Python package includes the MuJoCo library. For a standalone fresh environment, the basic commands are:

```bash
pip install mujoco
mjpython -m mujoco.viewer
```

On macOS, the passive viewer must run through `mjpython` to satisfy rendering/main-thread requirements. Plain MuJoCo installation is straightforward; reproducing the older LIBERO/robosuite dependency stack is a separate compatibility check. Pin that stack from the chosen example instead of blindly installing the newest MuJoCo. Sources: [installation/viewer](https://mujoco.readthedocs.io/en/stable/python.html), [macOS binaries](https://mujoco.readthedocs.io/en/stable/programming/index.html#getting-started).

Proposed split: Mac runs the simulation and viewer; a Linux Modal GPU in **`j999ucl`** serves π0.5; camera/state observations go to the policy server and action chunks return. Gemini receives selected observations and task outcomes, issues subtask instructions, and replans. The network bridge, latency, Mac LIBERO compatibility, model memory use and manipulation success are **not yet tested**. A working local simulator alone does not validate the full loop.

## Next concrete milestones and status boundary

1. Install a compatible isolated MuJoCo/LIBERO environment and open its supplied scene on the Mac.
2. Start a bounded π0.5 policy server on Modal, reproduce a supplied task, record actual success and latency, then stop compute.
3. Add Gemini orchestration and observable success/failure feedback.
4. Import premade rack assets and test one simplified maintenance interaction before expanding the scene.
5. Demonstrate an unscripted change with perception, decisions and physical recovery visible together.

This side conversation performed only read-only local inspection, documentation/source inspection and the Gemini model-list request before writing this note. **No MuJoCo installation, asset download, VLA inference, training, GPU launch or main-thread infrastructure cancellation was performed here.** Earlier Isaac/Modal setup belongs to the main conversation; this note does not establish its current status or cancel that work.
