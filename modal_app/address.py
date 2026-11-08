"""Print the running Modal sim's address:  python -m modal_app.address"""
import modal

print(modal.Dict.from_name("datacenter-robot-sim", create_if_missing=True).get("address")
      or "No sim running. Start one with: modal run --detach modal_app/sim.py")
