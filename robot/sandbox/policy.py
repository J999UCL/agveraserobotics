"""Local-first autopilot: settles clear-cut incidents on the robot, escalates the rest to Argon.

Decision rule per suspicious bay (all local, deterministic, milliseconds):
  * SMART bad AND at least one independent signal agrees (LED or thermal)  -> clear drive fault
      - no spares left              -> flag locally
      - RAID group rebuilding       -> wait locally (transient), up to MAX_WAITS
      - path obstructed             -> wait locally (transient), up to MAX_WAITS
      - otherwise                   -> swap locally, then verify
  * anything else (LED-only, heat-only, SMART bad with no corroboration)  -> escalate to Argon
  * any surprise during a local swap (jam, failed verification, condition that doesn't clear)
                                                                             -> escalate to Argon
Argon runs under the shared edge budget; if it is exhausted or unreachable the arm safe-stops.
"""

from __future__ import annotations

from agent import HOT_C, Budget, Trace, _log, local_alert, perception_table, pull_refusal, run

MAX_WAITS = 4


class Escalate(Exception):
    pass


def run_local_first(w, brain, budget: Budget | None = None, verbose: bool = True) -> Trace:
    budget = budget or Budget()
    trace = Trace()
    ctx = {"telemetry_read": set()}
    suspects = local_alert(w)
    severity = {"FAIL": 0, "WARN": 1, "OK": 2}
    suspects.sort(key=lambda i: severity.get(w.bays[i].smart, 2))
    history = [f"LOCAL MONITOR ALERT: suspicious bays {suspects}\n"
               f"controller:\n{w.rack_summary()}\nlocal perception:\n{perception_table(w)}"]

    def local(action, reason, fn, *args):
        result = fn(*args)
        trace.local_decisions += 1
        _log(trace, {"source": "local", "action": action, "args": list(args), "confidence": 1.0,
                     "hypothesis": reason}, result, verbose)
        history.append(f"[local] {action} {list(args)} -> {result}")
        return result

    def wait_while(cond, what):
        n = 0
        while cond() and n < MAX_WAITS:
            local("wait", f"{what} is transient; waiting", w.wait)
            n += 1
        if cond():
            raise Escalate(f"{what} did not clear after {MAX_WAITS} waits")

    for idx, bay in enumerate(suspects):
        if w.stopped:
            _flag_unhandled(w, suspects[idx:])
            break
        # Argon may have changed the rack since the alert: look again before acting on this bay.
        if bay not in local_alert(w) or any(f["bay"] == bay for f in w.flags):
            local("skip", "already resolved or flagged earlier in this incident", lambda b: "OK: skipped", bay)
            continue
        try:
            local("read_telemetry", "check drive health + RAID state", _tel(w, ctx), bay)
            v = w.perceive()[bay]
            smart_bad = w.bays[bay].smart != "OK"
            hot, led_bad = v.temp_c >= HOT_C, v.led in ("amber", "red")
            signals = f"smart={w.bays[bay].smart}, led={v.led}, thermal={v.temp_c:.0f}C"
            if not smart_bad:
                raise Escalate(f"drive health is OK but {signals}: not a clear drive fault")
            if not (hot or led_bad):
                raise Escalate(f"SMART bad but no corroborating LED/heat ({signals})")
            if w.spares <= 0:
                local("flag", "clear fault but no spare caddy", w.flag, bay,
                      f"drive fault ({signals}); no spares on site")
                continue
            wait_while(lambda: w.raid_state(bay) == "REBUILDING", "RAID rebuild")
            res = local("move_to", f"clear drive fault ({signals})", w.move_to, bay)
            if res.startswith("BLOCKED"):
                def blocked():
                    return w.move_to(bay).startswith("BLOCKED")
                wait_while(blocked, "obstruction")
            # Conditions may have changed during the waits: re-check right before the pull,
            # with the same fact checks the guard applies to Argon.
            wait_while(lambda: w.raid_state(bay) == "REBUILDING", "RAID rebuild")
            refusal = pull_refusal(w, bay)
            if refusal:
                raise Escalate(refusal)
            for action, fn in [("pull_caddy", w.pull_caddy), ("stow_old", w.stow_old),
                               ("pick_spare", w.pick_spare), ("insert_caddy", w.insert_caddy)]:
                res = local(action, "routine swap", fn)
                if not res.startswith("OK"):
                    raise Escalate(f"{action} returned: {res}")
            res = local("read_telemetry", "verify new drive", _tel(w, ctx), bay)
            if w.bays[bay].smart != "OK":
                raise Escalate(f"verification failed after insert: {res}")
        except Escalate as e:
            reason = str(e)
            trace.escalations.append(f"bay {bay}: {reason}")
            history.append(f"LOCAL AUTOPILOT -> ARGON: bay {bay}: {reason}. You now have control.")
            _log(trace, {"source": "local", "action": "ESCALATE", "args": {"bay": bay},
                         "hypothesis": reason}, "handed to Argon", verbose)
            run(w, brain, budget, verbose, history=history, ctx=ctx, trace=trace, focus=bay)
            if w.stopped:            # Argon (or the fallback) stopped the arm mid-incident
                _flag_unhandled(w, suspects[idx:])
                break
    return trace


def _flag_unhandled(w, bays):
    """Arm stopped before these suspects were settled: make sure a human hears about each."""
    flagged = {f["bay"] for f in w.flags}
    for bay in bays:
        if bay not in flagged:
            w.flag(bay, "arm stopped before this bay was resolved; needs review")


def _tel(w, ctx):
    def f(bay):
        ctx["telemetry_read"].add(bay)
        return w.telemetry(bay)
    return f
