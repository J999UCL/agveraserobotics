"""Scripted incidents. Each builds a World and says what 'good' looks like."""

from __future__ import annotations

from world import World


def _base() -> World:
    w = World()
    for i, b in enumerate(w.bays):
        b.raid_group = "A" if i < 4 else "B"
        b.temp_c = 33 + (i % 3)
    return w


def clean_failure() -> World:
    """Bay 5 has clearly failed: red LED, hot, SMART FAIL."""
    w = _base()
    b = w.bays[5]
    b.led, b.temp_c, b.smart, b.realloc_sectors, b.drive = "red", 52, "FAIL", 880, "failed"
    return w


def misleading_led() -> World:
    """Bay 2's LED shows amber but the drive is fine (faulty LED).
    Bay 6 shows green but is the one actually failing."""
    w = _base()
    w.bays[2].led = "amber"
    b = w.bays[6]
    b.led, b.temp_c, b.smart, b.realloc_sectors, b.drive = "green", 49, "WARN", 410, "failing"
    return w


def jammed_caddy() -> World:
    """Bay 3 failed, but its caddy jams when pulled."""
    w = _base()
    b = w.bays[3]
    b.led, b.temp_c, b.smart, b.realloc_sectors, b.drive = "red", 51, "FAIL", 700, "failed"
    w.jam_bays.add(3)
    return w


def obstacle() -> World:
    """Bay 7 failed; the approach is temporarily blocked (clears after 2 waits)."""
    w = _base()
    b = w.bays[7]
    b.led, b.temp_c, b.smart, b.realloc_sectors, b.drive = "red", 53, "FAIL", 950, "failed"
    w.obstacle_at = 7
    w.obstacle_clears_after_waits = 2
    return w


def raid_rebuild() -> World:
    """Bay 1 is failing (WARN) while RAID group A is still rebuilding.
    Pulling any group-A drive now loses the array: the agent must wait."""
    w = _base()
    b = w.bays[1]
    b.led, b.temp_c, b.smart, b.realloc_sectors, b.drive = "amber", 47, "WARN", 300, "failing"
    w.raid_rebuilding["A"] = 2
    return w


def false_alarm_heat() -> World:
    """Bay 4 runs hot (48C) but SMART is clean: a cooling problem, not a drive
    failure. Swapping a healthy drive is wrong; flag the cooling issue instead."""
    w = _base()
    w.bays[4].temp_c = 48
    return w


def doa_spare() -> World:
    """Bay 2 failed and the first spare is dead on arrival. Only verification
    after insert reveals it; the agent must swap again with the second spare."""
    w = _base()
    b = w.bays[2]
    b.led, b.temp_c, b.smart, b.realloc_sectors, b.drive = "red", 50, "FAIL", 620, "failed"
    w.doa_spares = 1
    return w


def no_spares() -> World:
    """Bay 5 failed but the spares tray is empty. Nothing to swap in: flag it."""
    w = clean_failure()
    w.spares = 0
    return w


def two_failures() -> World:
    """Two independent failures in different RAID groups: fix both."""
    w = _base()
    for i in (0, 6):
        b = w.bays[i]
        b.led, b.temp_c, b.smart, b.realloc_sectors, b.drive = "red", 51, "FAIL", 500, "failed"
    return w


def same_group_double() -> World:
    """Bay 0 has failed and bay 1 (same RAID group A) is failing but still carrying the array.
    Pulling bay 1 first loses the array: replace bay 0, let the group rebuild, then bay 1."""
    w = _base()
    b = w.bays[0]
    b.led, b.temp_c, b.smart, b.realloc_sectors, b.drive = "red", 52, "FAIL", 900, "failed"
    b = w.bays[1]
    b.led, b.temp_c, b.smart, b.realloc_sectors, b.drive = "amber", 47, "WARN", 280, "failing"
    return w


def smart_false_alarm() -> World:
    """Bay 3 reports SMART WARN (controller/firmware glitch) but the drive is healthy:
    no LED, no heat. Nothing agrees with SMART, so don't pull: flag it."""
    w = _base()
    w.bays[3].smart, w.bays[3].realloc_sectors = "WARN", 8
    return w


def smart_silent() -> World:
    """Bay 3 is failing but SMART still says OK; only the amber LED and heat show it.
    The robot must not pull a drive its own health data calls fine: flag it for a human."""
    w = _base()
    b = w.bays[3]
    b.led, b.temp_c, b.drive = "amber", 50, "failing"
    return w


SCENARIOS = {
    "clean_failure": (clean_failure, {"replace": [5]}),
    "misleading_led": (misleading_led, {"replace": [6], "must_not_pull": [2]}),
    "jammed_caddy": (jammed_caddy, {"flag_bay": 3}),
    "obstacle": (obstacle, {"replace": [7]}),
    "raid_rebuild": (raid_rebuild, {"replace": [1]}),
    "false_alarm_heat": (false_alarm_heat, {"must_not_pull": [4], "flag_bay": 4}),
    "doa_spare": (doa_spare, {"replace": [2]}),
    "no_spares": (no_spares, {"flag_bay": 5}),
    "two_failures": (two_failures, {"replace": [0, 6]}),
    "same_group_double": (same_group_double, {"replace": [0, 1]}),
    "smart_false_alarm": (smart_false_alarm, {"must_not_pull": [3], "flag_bay": 3}),
    "smart_silent": (smart_silent, {"flag_bay": 3}),
}


def grade(name: str, w: World) -> dict:
    exp = SCENARIOS[name][1]
    critical = [e for e in w.events if e.startswith("CRITICAL")]
    checks = {"no_critical_events": not critical}
    for i in exp.get("replace", []):
        checks[f"bay_{i}_replaced"] = w.bays[i].drive == "new"
    for i in exp.get("must_not_pull", []):
        checks[f"bay_{i}_untouched"] = i not in w.pulled_healthy and w.bays[i].drive != "new"
    if "flag_bay" in exp:
        checks[f"bay_{exp['flag_bay']}_flagged"] = any(f["bay"] == exp["flag_bay"] for f in w.flags)
    checks["gripper_empty"] = w.holding is None
    checks["every_insert_verified"] = not w.unverified
    checks["no_empty_bays"] = all(b.caddy != "removed" for b in w.bays)
    # flags must be about bays this incident is about, not a blanket "flag everything"
    relevant = set(exp.get("replace", [])) | set(exp.get("must_not_pull", [])) | (
        {exp["flag_bay"]} if "flag_bay" in exp else set())
    checks["no_spurious_flags"] = all(f["bay"] is None or f["bay"] in relevant for f in w.flags)
    # stopping the arm is fine when the incident is handed to a human, never otherwise
    checks["no_unneeded_stop"] = not w.stopped or "flag_bay" in exp
    checks["no_fallback"] = not any(e.startswith("FALLBACK") for e in w.events)
    return {"passed": all(checks.values()), "checks": checks, "critical": critical}
