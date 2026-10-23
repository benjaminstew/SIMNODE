"""Merge run_scaling_sweep JSONs into one, sorted by B.

Give either a directory (every *.json directly inside it is merged) or an explicit list of files:
    python -m data.bioreactor_simulation.merge_jsons --input_dir <dir> [--output <merged.json>]
    python -m data.bioreactor_simulation.merge_jsons --inputs a.json b.json --output <merged.json>
Files without a "results" block (i.e. not scaling-sweep outputs) are skipped. With --input_dir the
default output is <dir>/merged_<firstB>to<lastB>.json; the output file and any earlier merged_*.json
files in the directory are never read as inputs.
"""
import json
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
src = parser.add_mutually_exclusive_group(required=True)
src.add_argument("--input_dir", type=Path, metavar="DIR",
                 help="Directory whose *.json files (not subdirectories) are all merged.")
src.add_argument("--inputs", nargs="+", type=Path, metavar="JSON",
                 help="Explicit scaling-sweep .json files to merge.")
parser.add_argument("--output", type=Path, default=None, metavar="JSON",
                    help="Path to write the merged .json (default with --input_dir: "
                         "<dir>/merged_<firstB>to<lastB>.json; required with --inputs).")
args = parser.parse_args()

if args.input_dir is not None:
    in_dir = args.input_dir.expanduser().resolve()
    if not in_dir.is_dir():
        parser.error(f"--input_dir is not a directory: {in_dir}")
    # skip earlier merge outputs (merged_*.json) so re-running on the same directory doesn't re-ingest them
    files = sorted(f for f in in_dir.glob("*.json") if not f.name.startswith("merged_"))
else:
    if args.output is None:
        parser.error("--output is required with --inputs.")
    files = [Path(f).expanduser().resolve() for f in args.inputs]
out_path = args.output.expanduser().resolve() if args.output is not None else None

merged_results = {}
config = None
used, skipped, config_diffs = [], [], {}

for p in files:
    if out_path is not None and p == out_path:
        continue  # never merge a previous output into itself
    try:
        with open(p) as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as e:
        skipped.append(f"{p.name} (unreadable: {e})"); continue
    if not isinstance(data, dict) or not isinstance(data.get("results"), dict):
        skipped.append(f"{p.name} (no 'results' block)"); continue
    used.append(p.name)
    cfg = data.get("config", {})
    if config is None:
        config = dict(cfg)
    else:  # flag settings that differ from the first file (batch_sizes always differs, so ignore it)
        for k in set(config) | set(cfg):
            if k not in ("batch_sizes", "merged_from") and config.get(k) != cfg.get(k):
                config_diffs.setdefault(k, set()).add(p.name)
    # New schema: results[<B>][<repeat>] = {entry}. Merge at the (B, repeat) level, preferring
    # whichever repeat entry actually has data over an error stub.
    for B, reps in data["results"].items():
        dst = merged_results.setdefault(B, {})
        if not isinstance(reps, dict):
            continue
        for r, entry in reps.items():
            new_has_data = isinstance(entry, dict) and "error" not in entry
            existing = dst.get(r)
            existing_has_data = isinstance(existing, dict) and "error" not in existing
            if r not in dst or (new_has_data and not existing_has_data):
                dst[r] = entry

if not merged_results:
    raise SystemExit("No scaling-sweep results found to merge.")

# Sort by integer B, and each B's repeats by integer repeat index.
def _sorted_by_int(d):
    return {k: d[k] for k in sorted(d, key=lambda x: int(x))}

sorted_results = {B: (_sorted_by_int(merged_results[B]) if isinstance(merged_results[B], dict)
                      else merged_results[B])
                  for B in sorted(merged_results, key=lambda x: int(x))}
present = [int(B) for B in sorted_results]

config = config or {}
config["batch_sizes"] = present          # match the merged results, not the first file's range
config["merged_from"] = used

if out_path is None:
    out_path = in_dir / f"merged_{present[0]}to{present[-1]}.json"
out = {"config": config, "results": sorted_results}
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as fh:
    json.dump(out, fh, indent=2)

# ---- report ----
n_rep = config.get("num_training_repeats")
incomplete = {B: len(v) for B, v in sorted_results.items()
              if isinstance(v, dict) and n_rep is not None and
              sum(1 for e in v.values() if isinstance(e, dict) and "error" not in e) < n_rep}
gaps = sorted(set(range(present[0], present[-1] + 1)) - set(present))
print(f"Merged {len(used)} files -> {len(present)} B values: {present[0]}..{present[-1]}")
if gaps:
    print(f"  B values missing within that range: {gaps}")
if incomplete:
    print(f"  B values with fewer than {n_rep} successful repeats: {incomplete}")
if config_diffs:
    for k, names in sorted(config_diffs.items()):
        print(f"  WARNING: config '{k}' differs from {used[0]} in: {sorted(names)}")
if skipped:
    print(f"  Skipped: {skipped}")
print(f"Written to {out_path}")
