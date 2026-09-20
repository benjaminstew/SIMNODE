"""Merge multiple run_scaling_sweep JSONs into one, sorted by B."""
import json
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--inputs", nargs="+", type=Path, required=True, metavar="JSON",
                    help="Two or more scaling-sweep .json files to merge.")
parser.add_argument("--output", type=Path, required=True, metavar="JSON",
                    help="Path to write the merged .json.")
args = parser.parse_args()

files = args.inputs
out_path = args.output

merged_results = {}
config = None

for f in files:
    p = Path(f).expanduser().resolve()
    with open(p) as fh:
        data = json.load(fh)
    if config is None:
        config = data.get("config", {})
    for k, v in data.get("results", {}).items():
        if k not in merged_results:
            merged_results[k] = v
        else:
            # prefer whichever entry has more data (not just an error stub)
            existing_has_data = "error" not in merged_results[k]
            new_has_data = "error" not in v
            if new_has_data and not existing_has_data:
                merged_results[k] = v

# Sort by integer B
sorted_results = {k: merged_results[k]
                  for k in sorted(merged_results, key=lambda x: int(x))}

out = {"config": config or {}, "results": sorted_results}
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as fh:
    json.dump(out, fh, indent=2)

present = sorted(sorted_results.keys(), key=int)
print(f"Merged {len(present)} B values: {present[0]}..{present[-1]}")
print(f"Written to {out_path}")
