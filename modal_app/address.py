"""Print the address of the running Modal sim (for anyone on the same Modal workspace).

    python -m modal_app.address
"""

import modal

addr = modal.Dict.from_name("hotswap-sim", create_if_missing=True).get("address")
print(addr or "No sim running. Start one with: modal run modal_app/sim.py")
