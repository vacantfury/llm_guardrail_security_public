#!/usr/bin/env python3
"""Unit tests for the cluster health preflight (cluster_health.py).

No pytest — self-contained assert runner:  python tests/test_cluster_health.py
(exit 0 = all pass). `apply_health` is PURE (frozen-spec rewrite) and tested
directly; `probe_cluster`'s branching is tested by stubbing the module's `_ssh`
(no network, no cluster). Encodes the health policy (2026-07-19):
  unreachable / SLURM-down -> whole cluster dropped; xc Bedrock creds dead ->
  bedrock flipped OFF that cluster; api_keys is declared-only (never flipped).
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment import cluster_health as ch  # noqa: E402
from src.experiment.multi_cluster import ClusterSpec  # noqa: E402

AICR = ClusterSpec("aicr", "aicr", "~/r", "s", budget=2, max_submit=8,
                   bedrock=False, api_keys=True)
NURC = ClusterSpec("nurc", "nurc", "~/r", "s", budget=2, max_submit=8,
                   bedrock=False, api_keys=True)
XC = ClusterSpec("xc", "xc", "~/r", "s", budget=3, max_submit=8,
                 bedrock=True, api_keys=False)
POOL = [AICR, NURC, XC]


def _check(name, cond):
    if not cond:
        raise AssertionError(f"FAIL: {name}")
    print(f"  ok: {name}")


def _cp(returncode=0, stdout="", stderr=""):
    return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


# ---------------------------------------------------------------- apply_health

def test_apply_health_healthy_pool_untouched():
    healths = [
        ch.ClusterHealth("aicr", True, True, None, "ok"),
        ch.ClusterHealth("nurc", True, True, None, "ok"),
        ch.ClusterHealth("xc", True, True, True, "bedrock live"),
    ]
    eff, notes = ch.apply_health(POOL, healths)
    _check("all 3 kept", [c.name for c in eff] == ["aicr", "nurc", "xc"])
    _check("xc keeps bedrock", next(c for c in eff if c.name == "xc").bedrock)
    _check("no notes when healthy", notes == [])


def test_apply_health_drops_unreachable_cluster():
    healths = [
        ch.ClusterHealth("aicr", False, False, None, "UNREACHABLE — ssh failed"),
        ch.ClusterHealth("nurc", True, True, None, "ok"),
        ch.ClusterHealth("xc", True, True, True, "ok"),
    ]
    eff, notes = ch.apply_health(POOL, healths)
    _check("aicr dropped", [c.name for c in eff] == ["nurc", "xc"])
    _check("drop noted", any("aicr" in n and "DROPPED" in n for n in notes))


def test_apply_health_drops_slurm_down_cluster():
    healths = [
        ch.ClusterHealth("aicr", True, False, None, "SLURM not responding"),
        ch.ClusterHealth("nurc", True, True, None, "ok"),
        ch.ClusterHealth("xc", True, True, True, "ok"),
    ]
    eff, _ = ch.apply_health(POOL, healths)
    _check("slurm-down aicr dropped", [c.name for c in eff] == ["nurc", "xc"])


def test_apply_health_flips_bedrock_off_on_dead_creds():
    healths = [
        ch.ClusterHealth("aicr", True, True, None, "ok"),
        ch.ClusterHealth("nurc", True, True, None, "ok"),
        ch.ClusterHealth("xc", True, True, False, "bedrock creds EXPIRED"),
    ]
    eff, notes = ch.apply_health(POOL, healths)
    _check("xc still in pool (GPU still fine)", "xc" in [c.name for c in eff])
    _check("xc bedrock DISABLED", not next(c for c in eff if c.name == "xc").bedrock)
    _check("flip noted", any("xc" in n and "bedrock DISABLED" in n for n in notes))


def test_apply_health_unprobed_cluster_trusted():
    # A cluster with no health entry (not probed) is kept as declared.
    eff, notes = ch.apply_health(POOL, [])
    _check("all kept when unprobed", [c.name for c in eff] == ["aicr", "nurc", "xc"])
    _check("xc keeps declared bedrock", next(c for c in eff if c.name == "xc").bedrock)


def test_api_keys_never_flipped():
    # Even a fully-probed healthy xc keeps api_keys=False as DECLARED (health never
    # touches api_keys); aicr keeps api_keys=True.
    healths = [ch.ClusterHealth(c.name, True, True,
                                (True if c.bedrock else None), "ok") for c in POOL]
    eff, _ = ch.apply_health(POOL, healths)
    _check("aicr api_keys stays True", next(c for c in eff if c.name == "aicr").api_keys)
    _check("xc api_keys stays False", not next(c for c in eff if c.name == "xc").api_keys)


# ---------------------------------------------------------------- probe_cluster

def _stub_ssh(mapping, default=None):
    """Return a fake _ssh dispatching on a substring of the remote command."""
    def fake(spec, remote_cmd, timeout):
        for needle, cp in mapping.items():
            if needle in remote_cmd:
                return cp
        if default is not None:
            return default
        raise AssertionError(f"unexpected ssh cmd: {remote_cmd}")
    return fake


def _with_stub(mapping, fn, default=None):
    orig = ch._ssh
    ch._ssh = _stub_ssh(mapping, default)
    try:
        return fn()
    finally:
        ch._ssh = orig


def test_probe_unreachable():
    h = _with_stub({"true": _cp(returncode=255, stderr="Connection refused")},
                   lambda: ch.probe_cluster(XC, need_bedrock=True))
    _check("unreachable -> cluster_dead", h.cluster_dead)
    _check("unreachable detail", "UNREACHABLE" in h.detail)


def test_probe_slurm_down():
    h = _with_stub(
        {"true": _cp(0), "sinfo": _cp(returncode=127, stderr="sinfo: not found")},
        lambda: ch.probe_cluster(XC, need_bedrock=False))
    _check("reachable", h.reachable)
    _check("slurm down -> cluster_dead", h.cluster_dead and not h.slurm_up)


def test_probe_bedrock_live():
    h = _with_stub({
        "true": _cp(0),
        "sinfo": _cp(0, stdout="main\n"),
        "get-caller-identity": _cp(0, stdout='{"Account": "615299758675"}'),
    }, lambda: ch.probe_cluster(XC, need_bedrock=True))
    _check("healthy cluster not dead", not h.cluster_dead)
    _check("bedrock live", h.bedrock_live is True)


def test_probe_bedrock_expired():
    h = _with_stub({
        "true": _cp(0),
        "sinfo": _cp(0, stdout="main\n"),
        "get-caller-identity": _cp(returncode=254,
                                   stderr="An error occurred (ExpiredToken) ..."),
    }, lambda: ch.probe_cluster(XC, need_bedrock=True))
    _check("cluster itself alive", not h.cluster_dead)
    _check("bedrock dead on expiry", h.bedrock_live is False)
    _check("expiry detail mentions re-mint", "re-mint" in h.detail)


def test_probe_skips_bedrock_when_not_needed():
    # need_bedrock=False -> no sts call, bedrock_live stays None (tri-state).
    h = _with_stub({"true": _cp(0), "sinfo": _cp(0, stdout="main\n")},
                   lambda: ch.probe_cluster(XC, need_bedrock=False))
    _check("bedrock not probed -> None", h.bedrock_live is None)


def test_probe_non_bedrock_cluster_skips_sts():
    # aicr doesn't declare bedrock -> even with need_bedrock, no sts probe.
    h = _with_stub({"true": _cp(0), "sinfo": _cp(0, stdout="main\n")},
                   lambda: ch.probe_cluster(AICR, need_bedrock=True))
    _check("non-bedrock cluster -> bedrock_live None", h.bedrock_live is None)
    _check("non-bedrock cluster healthy", not h.cluster_dead)


def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    print(f"Running {len(tests)} cluster-health tests...")
    for t in tests:
        print(f"\n{t.__name__}:")
        t()
    print(f"\nALL {len(tests)} CLUSTER-HEALTH TESTS PASSED")


if __name__ == "__main__":
    main()
