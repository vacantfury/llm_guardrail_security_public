#!/usr/bin/env python3
"""Split ONE experiment preset across the cluster pool (AICR first, then NURC) and
give each cluster its own sub-preset. DRY-RUN by default: prints the split plan
and the exact ssh commands, and writes the sub-preset YAMLs locally — but
submits NOTHING. Pass --submit to actually place + sbatch each sub-preset over
ssh (sync the code to both clusters FIRST).

Usage:
    python dispatch.py autoattack_defense/experiment              # dry-run (plan only)
    python dispatch.py autoattack_defense/experiment --submit     # place + submit
    python dispatch.py <preset> --pool conf/cluster_pool.yaml
"""
import argparse
import sys
from pathlib import Path

from src.experiment.multi_cluster import DispatchError, dispatch
from src.utils.logger import get_logger

logger = get_logger(__name__)

CONF_DIR = Path(__file__).resolve().parent / "conf"


def _print_plan(result, submit: bool) -> None:
    plan = result.plan
    print(f"\n=== multi-cluster split: {result.preset_name} ===")
    if result.health_notes:
        print("\n--- cluster health preflight (routing around live state) ---")
        for note in result.health_notes:
            print(f"  {note}")
    print(f"distinct cluster-served models in preset: "
          f"{sorted(plan.total_cluster_models) or '(none — all API/no-model)'}")
    if not plan.total_cluster_models:
        print("NOTE: no GPU-served models — no vLLM servers to split. The "
              "assignment below still matters for CREDENTIALS: Bedrock tasks are "
              "routed to the Bedrock cluster (xc), non-Bedrock API tasks to an "
              "API-keys cluster (aicr/nurc).")

    active = [a for a in plan.assignments if a.active]
    for a in plan.assignments:
        tag = "" if a.active else "   (idle — nothing assigned)"
        print(f"\n[{a.cluster.name}] budget={a.cluster.budget} servers{tag}")
        if not a.active:
            continue
        print(f"  tasks:            {len(a.task_indices)}  "
              f"(indices {a.task_indices[:6]}{'...' if len(a.task_indices) > 6 else ''})")
        print(f"  servers to start: {sorted(a.server_models)}")
        print(f"  num_cluster_jobs: {a.num_cluster_jobs}  "
              f"(orchestrator + {len(a.server_models)} vLLM)")
        print(f"  sub-preset:       conf/experiment/{result.subpresets[a.cluster.name]}.yaml")

    if len(active) <= 1:
        print("\nNOTE: everything fit one cluster — routing sent it all to the "
              f"first capable cluster ({active[0].cluster.name if active else 'none'}) "
              "and the others stay idle. Expected when a matrix fits one cluster, "
              "or when its creds (Bedrock / API keys) pin it to a single cluster.")

    # Duplicated servers (a model served on >1 cluster because tasks split).
    served_on: dict[str, list[str]] = {}
    for a in active:
        for m in a.server_models:
            served_on.setdefault(m, []).append(a.cluster.name)
    dups = {m: cs for m, cs in served_on.items() if len(cs) > 1}
    if dups:
        print("\nDUPLICATED servers (served on >1 cluster due to the split; "
              "pin one to keep it single):")
        for m, cs in sorted(dups.items()):
            print(f"  {m}: {cs}")

    print("\n--- ssh commands " + ("(EXECUTED)" if submit else "(dry-run — not executed)") + " ---")
    for a in active:
        print(f"# {a.cluster.name}")
        for cmd in result.commands[a.cluster.name]:
            print(f"  {cmd}")

    if submit:
        print("\n--- submitted ---")
        for cname, out in result.submitted.items():
            print(f"  [{cname}] {out}")
    else:
        print("\nDRY-RUN: nothing submitted. Sub-preset YAMLs written locally "
              "for review. Re-run with --submit to place + sbatch them "
              "(sync the code to both clusters first).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-cluster experiment dispatcher")
    parser.add_argument("preset", help="preset name, e.g. autoattack_defense/experiment")
    parser.add_argument("--pool", default=str(CONF_DIR / "cluster_pool.yaml"),
                        help="cluster pool config (default: conf/cluster_pool.yaml)")
    parser.add_argument("--submit", action="store_true",
                        help="actually ssh + sbatch (default: dry-run). "
                             "Sync the code to both clusters first.")
    parser.add_argument("--no-probe", action="store_true",
                        help="skip the cluster health preflight (ssh reachability / "
                             "SLURM / Bedrock-creds liveness). Use offline or to "
                             "route purely on declared capabilities.")
    args = parser.parse_args()

    try:
        result = dispatch(args.preset, Path(args.pool), CONF_DIR,
                          submit=args.submit, probe=not args.no_probe)
    except (DispatchError, FileNotFoundError) as e:
        logger.error(str(e))
        sys.exit(1)

    _print_plan(result, args.submit)


if __name__ == "__main__":
    main()
