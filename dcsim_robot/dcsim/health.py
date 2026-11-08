"""Abstract server health model: what a BMC (iDRAC/iLO-style) would report.

Each physical component carries hidden health. The server reports per-slot readings
derived from whatever is currently installed, so pulling or inserting parts changes
the telemetry the agents see.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FAN_NOMINAL_RPM = 9000


@dataclass
class PartHealth:
    id: str
    kind: str  # "dimm" | "fan"
    serial: str
    status: str = "ok"  # ok | degraded | failed
    size_gb: int = 32
    ecc_ce: int = 0  # corrected errors
    ecc_ue: int = 0  # uncorrectable errors
    rpm: float = FAN_NOMINAL_RPM
    pending_post: float = 0.0  # seconds until a newly inserted part is verified
    fault_note: str = ""


@dataclass
class ServerHealth:
    name: str = "srv-r07-u08"
    parts: dict[str, PartHealth] = field(default_factory=dict)
    sel: list[dict] = field(default_factory=list)  # system event log
    t: float = 0.0

    def log(self, severity: str, msg: str, **kw) -> None:
        self.sel.append({"t": round(self.t, 1), "severity": severity, "message": msg, **kw})

    def inject(self, part_id: str, fault: str) -> None:
        p = self.parts[part_id]
        if fault == "dimm_uncorrectable":
            p.status, p.ecc_ue, p.ecc_ce, p.fault_note = "failed", 3, 412, "uncorrectable ECC errors"
        elif fault == "dimm_correctable":
            p.status, p.ecc_ce, p.fault_note = "degraded", 1874, "correctable ECC error rate above threshold"
        elif fault == "fan_failed":
            p.status, p.rpm, p.fault_note = "failed", 0.0, "fan stalled"
        elif fault == "fan_degraded":
            p.status, p.rpm, p.fault_note = "degraded", 3100.0, "fan speed far below target"
        else:
            raise ValueError(f"unknown fault {fault}")

    def update(self, dt: float, installed: dict[str, str | None], rng: np.random.Generator) -> list[dict]:
        """Advance time. `installed` maps tray slot -> part id (or None). Returns new events."""
        self.t += dt
        new = []
        for slot, pid in installed.items():
            if pid is None:
                continue
            p = self.parts[pid]
            if p.pending_post > 0:
                p.pending_post -= dt
                if p.pending_post <= 0:
                    ok = p.status == "ok"
                    msg = f"{slot}: {'part verified OK' if ok else 'new part reports a fault'} ({p.serial})"
                    self.log("info" if ok else "critical", msg, slot=slot, part=pid)
                    new.append({"event": "post_complete", "slot": slot, "part": pid, "ok": ok})
            if p.kind == "dimm" and p.status == "degraded" and rng.random() < dt * 2.0:
                p.ecc_ce += int(rng.integers(1, 6))
        return new

    # ------------------------------------------------------------------ report
    def report(self, installed: dict[str, str | None], tray_state: str) -> dict:
        dimms, fans = [], []
        for slot, pid in installed.items():
            p = self.parts.get(pid) if pid else None
            if slot.startswith("A"):
                if p is None:
                    dimms.append({"slot": slot, "present": False, "status": "absent"})
                else:
                    dimms.append({"slot": slot, "present": True, "serial": p.serial, "size_gb": p.size_gb,
                                  "ecc_corrected": p.ecc_ce, "ecc_uncorrectable": p.ecc_ue,
                                  "status": "verifying" if p.pending_post > 0 else p.status})
            else:
                if p is None:
                    fans.append({"slot": slot, "present": False, "rpm": 0, "status": "absent"})
                else:
                    fans.append({"slot": slot, "present": True, "serial": p.serial,
                                 "rpm": 0 if p.pending_post > 0 and p.status != "ok" else self._fan_rpm(p, installed),
                                 "status": "spinning up" if p.pending_post > 0 else p.status})
        working_fans = sum(1 for f in fans if f["status"] == "ok")
        missing = len(fans) - working_fans
        cpu = [round(52 + 7.5 * missing + d, 1) for d in (0.0, 1.8)]
        mem = sum(d["size_gb"] for d in dimms if d.get("present") and d["status"] != "failed")
        problems = [d for d in dimms if d["status"] not in ("ok", "verifying")] + \
                   [f for f in fans if f["status"] not in ("ok", "spinning up")]
        overall = "critical" if any(x["status"] == "failed" for x in problems) else ("warning" if problems else "ok")
        if any(t > 80 for t in cpu):
            overall = "critical"
        return {"server": self.name, "overall": overall, "tray": tray_state, "usable_memory_gb": mem,
                "cpu_temps_c": cpu, "dimms": dimms, "fans": fans}

    def _fan_rpm(self, p: PartHealth, installed: dict[str, str | None]) -> float:
        if p.status != "ok":
            return p.rpm
        n_bad = sum(1 for s, pid in installed.items() if s.startswith("F") and (pid is None or self.parts[pid].status != "ok"))
        return round(min(16000.0, FAN_NOMINAL_RPM * (1 + 0.25 * n_bad)))
