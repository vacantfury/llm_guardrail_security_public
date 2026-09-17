#!/usr/bin/env python3
"""Unit tests for the multi-cluster routing policy (multi_cluster.plan_split).

The project has no pytest setup, so this is a self-contained assert runner —
run it directly:  python tests/test_routing.py  (exit 0 = all pass)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment.multi_cluster import (  # noqa: E402
    ClusterSpec, TaskNeed, plan_split, DispatchError,
)


AICR = ClusterSpec("aicr", "aicr", "~/r", "s_aicr", budget=2, max_submit=8,
                   bedrock=False, api_keys=True)
NURC = ClusterSpec("nurc", "nurc", "~/r", "s_nurc", budget=2, max_submit=8,
                   bedrock=False, api_keys=True)
XC = ClusterSpec("xc", "xc", "~/r", "s_xc", budget=3, max_submit=8,
                 bedrock=True, api_keys=False)
POOL = [AICR, NURC, XC]


def _where(plan, idx):
    """Return the cluster name a task index landed on, or None if leftover."""
    for a in plan.assignments:
        if idx in a.task_indices:
            return a.cluster.name
    return None


def _check(name, cond):
    if not cond:
        raise AssertionError(f"FAIL: {name}")
    print(f"  ok: {name}")


def test_gpu_only_prefers_first_pool_cluster():
    # One GPU model, no API creds needed -> aicr (first in pool), not xc.
    plan = plan_split([TaskNeed(0, frozenset({"qwen"}))], POOL, {})
    _check("gpu-only -> aicr (pool first)", _where(plan, 0) == "aicr")
    _check("gpu-only not leftover", not plan.leftover)


def test_gpu_overflow_spills_to_xc_last():
    # 6 distinct-model GPU tasks: aicr(2) + nurc(2) fill, remainder spills to xc.
    needs = [TaskNeed(i, frozenset({f"m{i}"})) for i in range(6)]
    plan = plan_split(needs, POOL, {})
    _check("no leftover (fits 2+2+3=7)", not plan.leftover)
    placed = {c: [] for c in ("aicr", "nurc", "xc")}
    for i in range(6):
        placed[_where(plan, i)].append(i)
    _check("aicr took 2", len(placed["aicr"]) == 2)
    _check("nurc took 2", len(placed["nurc"]) == 2)
    _check("xc took the overflow 2 (third tier)", len(placed["xc"]) == 2)


def test_bedrock_task_goes_to_xc_only():
    plan = plan_split([TaskNeed(0, frozenset(), needs_bedrock=True)], POOL, {})
    _check("bedrock -> xc", _where(plan, 0) == "xc")


def test_bedrock_plus_gpu_goes_to_xc():
    # Needs Bedrock (target) AND a GPU judge -> only xc can do both.
    plan = plan_split(
        [TaskNeed(0, frozenset({"wildguard"}), needs_bedrock=True)], POOL, {})
    _check("bedrock+gpu -> xc", _where(plan, 0) == "xc")
    xc_asg = next(a for a in plan.assignments if a.cluster.name == "xc")
    _check("xc serves the gpu judge", "wildguard" in xc_asg.server_models)


def test_other_api_never_routes_to_xc():
    plan = plan_split([TaskNeed(0, frozenset(), needs_other_api=True)], POOL, {})
    _check("other-api -> aicr (has keys)", _where(plan, 0) == "aicr")


def test_gpu_with_other_api_excludes_xc():
    needs = [TaskNeed(i, frozenset({f"m{i}"}), needs_other_api=True) for i in range(4)]
    plan = plan_split(needs, POOL, {})
    for i in range(4):
        _check(f"task {i} not on xc (no API-keys)", _where(plan, i) in ("aicr", "nurc"))
    # 4 tasks, aicr(2)+nurc(2) exactly fills; xc can't take any -> the 5th leftover.
    plan5 = plan_split(needs + [TaskNeed(4, frozenset({"m4"}), needs_other_api=True)],
                       POOL, {})
    _check("5th other-api task leftover (xc can't help)", plan5.leftover == [4])


def test_bedrock_plus_other_api_unsatisfiable():
    plan = plan_split(
        [TaskNeed(0, frozenset(), needs_bedrock=True, needs_other_api=True)], POOL, {})
    _check("bedrock+other-api is leftover", plan.leftover == [0])
    _check("leftover reason mentions both",
           "both" in plan.leftover_reasons[0].lower())


def test_pin_forces_cluster():
    plan = plan_split([TaskNeed(0, frozenset({"big_judge"}))], POOL,
                      {"big_judge": "nurc"})
    _check("pinned to nurc", _where(plan, 0) == "nurc")


def test_pin_to_incapable_cluster_errors():
    # Pin a bedrock task to aicr (no Bedrock creds) -> loud error.
    try:
        plan_split([TaskNeed(0, frozenset({"m"}), needs_bedrock=True)], POOL,
                   {"m": "aicr"})
    except DispatchError as e:
        _check("pin-to-incapable raises", "lacks the credentials" in str(e))
        return
    raise AssertionError("FAIL: expected DispatchError for incapable pin")


def test_pure_api_no_creds_needed_lands_first():
    # A task with no models at all (pure no-op / API with no served model) still
    # places on the first pool cluster.
    plan = plan_split([TaskNeed(0, frozenset())], POOL, {})
    _check("empty-need -> aicr", _where(plan, 0) == "aicr")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} routing tests...")
    for t in tests:
        print(f"\n{t.__name__}:")
        t()
    print(f"\nALL {len(tests)} ROUTING TESTS PASSED")


if __name__ == "__main__":
    main()
