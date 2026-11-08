"""Reasoning agent for the toy rack.

Split of responsibilities (this is what the write-up describes):
  * local perception - LED / heat / obstruction read from image pixels (perception.py)
  * local reflexes   - proximity / force limits inside the world (world.py / mujoco_adapter.py)
  * local autopilot  - settles clear-cut incidents itself (policy.py)
  * local guard      - checks on irreversible actions + edge budget (this file)
  * Argon            - called only when the autopilot escalates: conflicting evidence,
                       jams, failed verification, conditions that don't clear

Brain: Gemini 4 Argon, stateless, one JSON action per call.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field

from world import World

MODEL = "gemini-4-argon"

SYSTEM = """You are the reasoning brain of a robotic arm that services an unmanned data-centre rack
(8 drive bays, 0-7). A local autopilot handles routine swaps; when it cannot settle an incident
safely it hands it to you with its reason. No human is available to ask. Decide yourself;
safety beats speed.

Each turn, reply with ONE JSON object and nothing else:
{"observation": "<what the latest evidence shows>",
 "hypothesis": "<what you believe is wrong, and why>",
 "confidence": <0.0-1.0 that your hypothesis is right>,
 "action": "<one action name>",
 "args": {...}}

Actions:
  look                         -> front camera image (LED colours, caddy state, arm, obstructions) + thermal image
  read_telemetry {"bay": n}    -> SMART, reallocated sectors, caddy and RAID group state for one bay
                                  (NO temperature: heat is only visible on the thermal image)
  move_to {"bay": n}           -> move arm to a bay (proximity reflex may block it)
  wait                         -> wait 60 s and re-check conditions
  pull_caddy                   -> pull the caddy at the arm's bay (IRREVERSIBLE: pulling a healthy drive or
                                  pulling during a RAID rebuild destroys data)
  stow_old                     -> drop the pulled caddy in the RMA bin
  pick_spare                   -> take a new caddy from the spares tray
  insert_caddy                 -> seat the new caddy in the arm's (empty) bay
  flag {"bay": n, "reason": s} -> leave a note for the next human visit
  safe_stop {"reason": s}      -> stop the arm and hold
  done {"summary": s}          -> incident finished

Rules:
- Corroborate before acting: an LED alone can be wrong, and so can heat alone (cooling fault).
  A drive needs replacing when its own health data (SMART) is bad and another signal agrees.
- Never pull a drive while its RAID group is REBUILDING or if it might be healthy.
- If a reflex reports BLOCKED or JAMMED, do not force it.
- When your evidence conflicts, gather more before acting; say so in the hypothesis.
- ONLY obstructions and RAID rebuilds are transient: wait and re-check (up to 4 waits) before
  giving up. A jam, a dead new drive or a sensor conflict will not fix itself by waiting.
- A newly inserted drive that reports smart=FAIL is dead on arrival: pull it, stow it, insert
  another spare if one is left, then verify again.
- A failed drive left in place is a continuing risk, so fix it yourself if you safely can;
  flag + safe_stop is the last resort. Non-drive faults (bad LED, cooling) are flagged, not swapped.
- After every insert, verify with read_telemetry (smart=OK) before done; a camera look is not
  verification. "done" is refused while a drive is held or an insert is unverified.
- Never pull a drive while another member of its RAID group is failed or out: replace the dead
  one first and let the group rebuild. A new drive starts a rebuild: wait for it.
"""

IRREVERSIBLE = {"pull_caddy"}
MIN_CONFIDENCE_IRREVERSIBLE = 0.8


@dataclass
class Budget:
    max_model_calls: int = 24
    max_seconds: float = 900.0       # wall clock for the WHOLE incident, across escalations
    max_call_seconds: float = 120.0  # one model call is abandoned after this


@dataclass
class Trace:
    steps: list[dict] = field(default_factory=list)
    model_calls: int = 0
    model_seconds: float = 0.0
    local_decisions: int = 0
    escalations: list[str] = field(default_factory=list)
    fallback: str | None = None
    started: float | None = None     # set on the first Argon loop; shared by later escalations


HOT_C = 45.0


def local_alert(w: World) -> list[int]:
    """Local monitor: bad SMART from the controller, or LED / heat anomalies from local perception."""
    views = w.perceive()
    return [i for i, b in enumerate(w.bays)
            if b.caddy != "removed" and (b.smart != "OK" or views[i].temp_c >= HOT_C
                                         or views[i].led in ("amber", "red", "unknown"))]


def perception_table(w: World) -> str:
    return "\n".join(f"bay {v.bay}: led={v.led} thermal={v.temp_c:.0f}C"
                     + (" OBSTRUCTED" if v.obstructed else "") for v in w.perceive())


def execute(w: World, action: str, args: dict, ctx: dict) -> tuple[str, list[bytes]]:
    imgs: list[bytes] = []
    a = action
    if a == "look":
        imgs = [w.camera_png(), w.thermal_png()]
        # Images reach the model once; this text stays in the history after that.
        # A look does NOT verify a new drive: only telemetry reading smart=OK does.
        seen = "; ".join(perception_table(w).splitlines())
        return f"camera + thermal images attached. local perception read: {seen}", imgs
    if a == "read_telemetry":
        bay = int(args["bay"])
        ctx["telemetry_read"].add(bay)
        return w.telemetry(bay), imgs
    if a == "move_to":
        return w.move_to(int(args["bay"])), imgs
    if a == "wait":
        return w.wait(), imgs
    if a == "pull_caddy":
        return w.pull_caddy(), imgs
    if a == "stow_old":
        return w.stow_old(), imgs
    if a == "pick_spare":
        return w.pick_spare(), imgs
    if a == "insert_caddy":
        return w.insert_caddy(), imgs
    if a == "flag":
        return w.flag(args.get("bay"), str(args.get("reason", ""))), imgs
    if a == "safe_stop":
        return w.safe_stop(str(args.get("reason", ""))), imgs
    return f"ERROR: unknown action {a!r}", imgs


def pull_refusal(w: World, bay: int) -> str | None:
    """Facts that must hold, right now, before any pull (model or local). Refusal or None.

    Checked against live controller state and fresh perception, never against what was
    read earlier, so a rebuild that started after the last telemetry read still blocks.
    """
    b = w.bays[bay]
    if b.caddy == "removed":
        return None                      # the world reports the empty bay itself
    if w.raid_state(bay) == "REBUILDING":
        return (f"POLICY: pull refused, raid group {b.raid_group} is REBUILDING; "
                "pulling now loses the array. Wait and re-check.")
    if b.smart == "OK":
        return f"POLICY: pull refused, bay {bay} reports smart=OK; the drive may be healthy."
    down = [j for j, x in enumerate(w.bays) if j != bay and x.raid_group == b.raid_group
            and (x.caddy == "removed" or x.smart == "FAIL")]
    if down and b.smart != "FAIL":
        return (f"POLICY: pull refused, raid group {b.raid_group} member(s) {down} already down; "
                f"bay {bay} (smart={b.smart}) is still carrying the array. Replace those first "
                "and let the group rebuild.")
    v = w.perceive()[bay]
    if not (v.led in ("amber", "red") or v.temp_c >= HOT_C):
        return (f"POLICY: pull refused, smart={b.smart} at bay {bay} but no corroborating "
                f"LED or heat (led={v.led}, thermal={v.temp_c:.0f}C). Gather more evidence, or flag.")
    return None


def guard(w: World, decision: dict, ctx: dict) -> str | None:
    """Local policy check on irreversible actions and on 'done'. Returns a refusal or None."""
    if decision["action"] == "done":
        if w.holding:
            return f"POLICY: done refused, gripper still holding a {w.holding} caddy."
        pending = w.unverified - {f["bay"] for f in w.flags}
        if pending:
            return (f"POLICY: done refused, inserted drive(s) in bay(s) {sorted(pending)} not "
                    "verified healthy (read_telemetry must show smart=OK). Verify, re-swap, or flag.")
        return None
    if decision["action"] not in IRREVERSIBLE:
        return None
    conf = float(decision.get("confidence", 0))
    if conf < MIN_CONFIDENCE_IRREVERSIBLE:
        return (f"POLICY: pull refused, confidence {conf:.2f} < {MIN_CONFIDENCE_IRREVERSIBLE}. "
                "Gather more evidence, or flag.")
    if w.arm_at == "home":
        return None
    bay = int(w.arm_at)
    if bay not in ctx["telemetry_read"]:
        return f"POLICY: pull refused, telemetry for bay {bay} not read this incident."
    return pull_refusal(w, bay)


def make_safe_and_stop(w: World, reason: str, alert: list[int]) -> None:
    """Local fallback when the brain is gone: leave nothing in the gripper, stop, flag everything open."""
    if w.holding == "old":
        w.stow_old()                     # failed drive goes to the RMA bin, not left in the gripper
    elif w.holding in ("new", "new_doa"):
        bay = None if w.arm_at == "home" else int(w.arm_at)
        if bay is not None and w.bays[bay].caddy == "removed":
            w.insert_caddy()             # restores redundancy; flagged below as unverified
        else:
            w.return_spare()
    w.events.append(f"FALLBACK: {reason}")
    w.safe_stop(reason)
    open_bays = set(alert) | set(local_alert(w)) | set(w.unverified) | {
        i for i, b in enumerate(w.bays) if b.caddy == "removed"}
    flagged = {f["bay"] for f in w.flags}
    for bay in sorted(open_bays - flagged):
        note = "inserted, NOT verified" if bay in w.unverified else (
            "bay empty" if w.bays[bay].caddy == "removed" else "unresolved fault")
        w.flag(bay, f"{reason}; {note}; needs review")
    if not open_bays and not w.flags:
        w.flag(None, f"{reason}; needs review")


# --------------------------------------------------------------------- brains
class ArgonBrain:
    def __init__(self, thinking: str = "LOW"):
        from google import genai
        from google.genai import types
        import os
        self.types = types
        # Only Google/Gemini-named variables: never pick up some other service's *_API_KEY.
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or next(
            (v for k, v in os.environ.items() if "GEMINI" in k.upper()), None)
        if not key:
            raise RuntimeError("no Gemini key: set GEMINI_API_KEY (or GOOGLE_API_KEY)")
        self.client = genai.Client(api_key=key)
        self.thinking = thinking

    def decide(self, prompt: str, images: list[bytes]) -> tuple[dict, str]:
        t = self.types
        parts = [t.Part.from_text(text=prompt)]
        for png in images:
            parts.append(t.Part.from_bytes(data=png, mime_type="image/png"))
        resp = self.client.models.generate_content(
            model=MODEL,
            contents=[t.Content(role="user", parts=parts)],
            config=t.GenerateContentConfig(
                system_instruction=SYSTEM,
                response_mime_type="application/json",
                thinking_config=t.ThinkingConfig(thinking_level=self.thinking),
            ),
        )
        raw = resp.text or ""
        return parse_json(raw), raw


def parse_json(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {"action": "safe_stop", "args": {"reason": "unparseable model output"},
                "confidence": 0.0, "observation": raw[:200], "parse_error": True}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"action": "safe_stop", "args": {"reason": "invalid JSON from model"},
                "confidence": 0.0, "observation": raw[:200], "parse_error": True}


MAX_PARSE_RETRIES = 1


class CallTimeout(Exception):
    pass


def _call_with_deadline(brain, prompt: str, images: list[bytes], seconds: float):
    """Run brain.decide in a daemon thread; give up after `seconds` (the call can't be killed,
    but the arm no longer waits on it)."""
    box: dict = {}

    def target():
        try:
            box["out"] = brain.decide(prompt, images)
        except BaseException as e:  # noqa: BLE001 - re-raised in the caller's thread
            box["err"] = e

    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(max(seconds, 0.0))
    if th.is_alive():
        raise CallTimeout(f"no answer within {seconds:.1f}s")
    if "err" in box:
        raise box["err"]
    return box["out"]


# ----------------------------------------------------------------------- loop
def run(w: World, brain, budget: Budget | None = None, verbose: bool = True,
        history: list[str] | None = None, ctx: dict | None = None,
        trace: Trace | None = None, focus: int | None = None) -> Trace:
    """Argon loop: standalone (Argon-only mode) or continuing an escalated incident."""
    budget = budget or Budget()
    trace = trace or Trace()
    ctx = ctx if ctx is not None else {"telemetry_read": set()}
    alert = local_alert(w)
    if focus is not None:
        alert = [focus] + [a for a in alert if a != focus]
    if history is None:
        history = [f"LOCAL MONITOR ALERT: suspicious bays {alert}\n{w.rack_summary()}"]
    pending_images: list[bytes] = []
    if trace.started is None:
        trace.started = time.time()
    t0 = trace.started               # one clock for the whole incident, not per escalation
    parse_failures = 0

    while True:
        elapsed = time.time() - t0
        if trace.model_calls >= budget.max_model_calls or elapsed > budget.max_seconds:
            trace.fallback = "edge budget exhausted -> local fallback: make safe, safe_stop + flag"
            make_safe_and_stop(w, "agent budget exhausted before resolution", alert)
            _log(trace, {"action": "LOCAL_FALLBACK"}, trace.fallback, verbose)
            break

        prompt = ("Incident log so far (oldest first):\n" + "\n".join(_window(history)) +
                  f"\n\nArm at: {w.arm_at}; holding: {w.holding}; spares: {w.spares}; "
                  f"model calls left: {budget.max_model_calls - trace.model_calls}.\nNext action?")
        tc = time.time()
        deadline = min(budget.max_call_seconds, budget.max_seconds - elapsed)
        try:
            decision, raw = _call_with_deadline(brain, prompt, pending_images, deadline)
        except Exception as e:  # network / quota / hang: degrade locally, don't crash the arm
            trace.fallback = (f"model unreachable ({type(e).__name__}: {e}) -> "
                              "local fallback: make safe, safe_stop + flag")
            make_safe_and_stop(w, "model unreachable", alert)
            _log(trace, {"action": "LOCAL_FALLBACK"}, trace.fallback, verbose)
            break
        dt = time.time() - tc
        trace.model_calls += 1
        trace.model_seconds += dt

        if decision.get("parse_error") and parse_failures < MAX_PARSE_RETRIES:
            parse_failures += 1      # one bad reply is a retry, not a reason to stop the arm
            result = "FORMAT ERROR: reply with exactly ONE JSON object as specified. Try again."
            _log(trace, decision, result, verbose, dt)
            history.append(f"[{trace.model_calls}] you: <unparseable reply> -> {result}")
            continue                 # keep pending_images: the model never used them
        parse_failures = 0
        pending_images = []

        action = decision.get("action", "")
        args = decision.get("args") or {}
        refusal = guard(w, decision, ctx)
        if refusal:
            result = refusal
        elif action == "done":
            _log(trace, decision, "DONE", verbose, dt)
            break
        else:
            try:
                result, pending_images = execute(w, action, args, ctx)
            except Exception as e:
                result = f"ERROR: {e}"
        _log(trace, decision, result, verbose, dt)
        history.append(f"[{trace.model_calls}] you: {action} {json.dumps(args)} "
                       f"(conf {decision.get('confidence')}) -> {result}")
        if action == "safe_stop" and not refusal:
            break
    return trace


HISTORY_WINDOW = 30


def _window(history: list[str]) -> list[str]:
    """Last entries, but never drop the opening alert (the incident's ground facts)."""
    if len(history) <= HISTORY_WINDOW:
        return history
    keep = HISTORY_WINDOW - 2
    return [history[0], f"... ({len(history) - 1 - keep} older entries omitted) ..."] + history[-keep:]


def _log(trace: Trace, decision: dict, result: str, verbose: bool, dt: float = 0.0):
    src = decision.get("source") or ("local" if decision.get("action") == "LOCAL_FALLBACK" else "argon")
    step = {"source": src, "decision": decision, "result": result, "model_s": round(dt, 1)}
    trace.steps.append(step)
    if verbose:
        print(f"  [{len(trace.steps):02d}] {src:<5} {decision.get('action')} {decision.get('args', {})} "
              f"conf={decision.get('confidence')} ({dt:.0f}s)\n"
              f"       why: {str(decision.get('hypothesis', ''))[:160]}\n"
              f"       -> {result}", flush=True)
