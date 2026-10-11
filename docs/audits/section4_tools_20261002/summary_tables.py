"""Prompt 4 step 4 helper: markdown tables (per-model status, per-stream coverage, classes per model) from the
immutable reports listed in docs/audits/section4_parity_manifest_<date>.json. Prints markdown to stdout."""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATE = sys.argv[1] if len(sys.argv) > 1 else "20261005"
manifest = json.loads((ROOT / f"docs/audits/section4_parity_manifest_{DATE}.json").read_text(encoding="utf-8"))
reports = [json.loads((ROOT / r["path"]).read_text(encoding="utf-8")) | {"_entry": r} for r in manifest["reports"]]

print("| ID | Model | Stream | Verdict | Fail classes (cells/checks) | Comparable endpoints | Decision flips | Report sha256 (12) |")
print("|---:|---|---|---|---|---:|---:|---|")
for r in reports:
    m = r["model"]
    classes = defaultdict(int)
    for g in r["mismatches"]:
        top = g["cls"].split("/")[0]
        if top in {"training_side_noncausal", "app_producer_bug", "unexplained"}:
            classes[g["cls"]] += g["count"]
    comparable = sum(w["comparable_endpoints"] for w in r["windows"])
    endpoints = sum(w["endpoints"] for w in r["windows"])
    cls = "<br>".join(f"{k} ({v:,})" for k, v in sorted(classes.items())) or "none"
    print(f"| {m['queue_id']} | `{m['model_id']}` | {m['stream']} | **{r['verdict']}** | {cls} | {comparable:,}/{endpoints:,} | "
          f"{len(r['decision_flips'])} | `{r['_entry']['sha256'][:12]}` |")

for stream in ("ES", "EURUSD"):
    rs = [r for r in reports if r["model"]["stream"] == stream]
    scenarios = list(rs[0]["coverage"])
    print(f"\n**{stream} coverage** (E = exercised with comparable endpoints for that model, - = not)\n")
    print("| Scenario | " + " | ".join(str(r["model"]["queue_id"]) for r in rs) + " | Segments |")
    print("|---|" + "---|" * len(rs) + "---|")
    for s in scenarios:
        segs = sorted({x for r in rs for x in r["coverage"][s].get("segments", [])})
        print(f"| {s} | " + " | ".join("E" if r["coverage"][s]["status"] == "exercised" else "-" for r in rs) + f" | {', '.join(segs) or rs[0]['coverage'][s]['reason']} |")
    print(f"| S2 setup events | " + " | ".join(r["setup_events"]["status"].split("_")[0] for r in rs) + " | ES: H windows only; EURUSD: D5 |")
    print(f"| Source parity (S) | " + " | ".join(r["source_parity"]["status"] for r in rs) + " | protocol fallback |")
