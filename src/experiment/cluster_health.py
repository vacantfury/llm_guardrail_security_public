"""Runtime liveness probe for the cluster pool — the health layer in FRONT of the
pure `plan_split` router (multi_cluster.py).

WHY. A cluster's DECLARED capabilities (conf/clusters/*.yaml: `bedrock`, `api_keys`,
plus implicit GPU serving) are static, but availability is not. Two things go stale
on their own schedule and would otherwise fail a run only AFTER it launches:

So BEFORE routing we probe each pool cluster for the capabilities THIS run needs,
flip the dead ones off the (frozen) `ClusterSpec`, and let the existing pure router
place work only where it can actually run — aborting early with an actionable fix
("re-mint the creds") when a needed capability is dead everywhere, instead of
grinding a launched matrix into per-cell failures.

DESIGN. This module is the ONLY I/O layer (ssh); `plan_split` stays pure and
unit-testable. `probe_cluster` NEVER raises — a probe failure is just "dead" with a
human `detail`. `apply_health` is pure (frozen-spec rewrite), so it is unit-tested.

CAPABILITY POLICY (fail-closed vs fail-open):
  * reachability / SLURM  → fail CLOSED (a real check decides; unreachable = dropped).
  * bedrock creds         → fail CLOSED (a reliable `sts` check; expiry is the whole
                            point). A probe that can't run at all is treated as dead.
  * api_keys              → declared-only (ASSUME-LIVE) in v1: there is no cheap,
                            reliable liveness probe (keys are injected at job
                            launch, not present in a login shell) and it has never
                            been the failure mode. A false "keys dead" would wrongly
                            strand aicr/nurc — worse than the rare miss.
"""
from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Optional

# Keep the whole preflight snappy: an unreachable cluster should cost seconds, not
# a hung ssh. Bedrock's `sts` call gets a bit more headroom (a real AWS round-trip).
_SSH_CONNECT_TIMEOUT = 8
_STS_EXTRA = 12


@dataclass
class ClusterHealth:
    """Liveness verdict for one pool cluster, for the capabilities a run needs.

    `bedrock_live` is tri-state: True/False when probed, None when not probed
    (the run doesn't need Bedrock, or the cluster doesn't declare it).
    """
    name: str
    reachable: bool
    slurm_up: bool
    bedrock_live: Optional[bool]
    detail: str  # one-line human summary: what's up/down + the fix

    @property
    def cluster_dead(self) -> bool:
        """Whole cluster unusable this run (drop it from the pool)."""
        return not (self.reachable and self.slurm_up)


def _ssh(spec, remote_cmd: str, timeout: int) -> subprocess.CompletedProcess:
    """One ssh round-trip. Caller handles TimeoutExpired / OSError."""
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes",
         "-o", f"ConnectTimeout={_SSH_CONNECT_TIMEOUT}", spec.ssh, remote_cmd],
        capture_output=True, text=True, timeout=timeout)


def _probe_bedrock(spec, timeout: int) -> tuple[bool, str]:
    """Is Bedrock invocable on this cluster right now? Sources the cluster's AWS env
    (xc's `~/.xc_env` sets the profile + shared-creds path) then runs `sts`.
    ExpiredToken / any non-identity result → dead (fail-closed)."""
    cmd = "source ~/.xc_env 2>/dev/null; aws sts get-caller-identity 2>&1"
    try:
        r = _ssh(spec, cmd, timeout + _STS_EXTRA)
    except (subprocess.TimeoutExpired, OSError) as e:
        return False, f"bedrock probe could not run ({type(e).__name__})"
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode == 0 and '"Account"' in out:
        return True, "bedrock creds LIVE"
    if "ExpiredToken" in out or "expired" in out.lower():
        return False, "bedrock creds EXPIRED — box operator must re-mint on the box (see runbook), then rerun"
    last = out.strip().splitlines()[-1][:140] if out.strip() else "unknown error"
    return False, f"bedrock creds unavailable — {last}"


def probe_cluster(spec, need_bedrock: bool,
                  timeout: int = _SSH_CONNECT_TIMEOUT + 4) -> ClusterHealth:
    """Probe ONE cluster for the capabilities a run needs. Never raises."""
    # 1. reachability — a cheap `true` over ssh.
    try:
        rc = _ssh(spec, "true", timeout)
    except (subprocess.TimeoutExpired, OSError) as e:
        return ClusterHealth(spec.name, False, False, None,
                             f"UNREACHABLE — ssh {spec.ssh} failed ({type(e).__name__})")
    if rc.returncode != 0:
        return ClusterHealth(spec.name, False, False, None,
                             f"UNREACHABLE — ssh {spec.ssh} exit {rc.returncode}: "
                             f"{(rc.stderr or '').strip()[:120]}")

    # 2. SLURM responsive — `sinfo` must return at least one partition line.
    try:
        sr = _ssh(spec, "sinfo -h -o '%P' 2>/dev/null", timeout)
        slurm_up = sr.returncode == 0 and bool((sr.stdout or "").strip())
    except (subprocess.TimeoutExpired, OSError):
        slurm_up = False
    if not slurm_up:
        return ClusterHealth(spec.name, True, False, None,
                             "reachable but SLURM not responding (sinfo failed) — "
                             "cluster down or misconfigured")

    # 3. bedrock — only where the run needs it AND the cluster declares it.
    bedrock_live: Optional[bool] = None
    bmsg = ""
    if need_bedrock and spec.bedrock:
        bedrock_live, bmsg = _probe_bedrock(spec, timeout)

    bits = ["reachable", "SLURM up"]
    if bedrock_live is not None:
        bits.append(bmsg)
    return ClusterHealth(spec.name, True, True, bedrock_live,
                         f"{spec.name}: " + "; ".join(bits))


def probe_pool(clusters, need_bedrock: bool) -> list[ClusterHealth]:
    """Probe every pool cluster concurrently (independent ssh round-trips)."""
    if not clusters:
        return []
    with ThreadPoolExecutor(max_workers=min(8, len(clusters))) as ex:
        return list(ex.map(lambda c: probe_cluster(c, need_bedrock), clusters))


def apply_health(clusters, healths) -> tuple[list, list[str]]:
    """Fold liveness into the pool. PURE (frozen-spec rewrite → unit-testable).

    Returns (effective_clusters, notes):
      * a cluster that is unreachable / SLURM-down is DROPPED;
      * a live cluster whose declared `bedrock` probed dead gets `bedrock=False`;
      * `notes` are human one-liners for every drop / capability flip (empty when
        the whole pool is healthy).
    api_keys is never flipped here (declared-only policy — see module docstring).
    """
    hmap = {h.name: h for h in healths}
    effective, notes = [], []
    for c in clusters:
        h = hmap.get(c.name)
        if h is None:              # not probed → trust the declaration
            effective.append(c)
            continue
        if h.cluster_dead:
            notes.append(f"[{c.name}] DROPPED — {h.detail}")
            continue
        newc = c
        if c.bedrock and h.bedrock_live is False:
            newc = replace(newc, bedrock=False)
            notes.append(f"[{c.name}] bedrock DISABLED — {h.detail}")
        effective.append(newc)
    return effective, notes
