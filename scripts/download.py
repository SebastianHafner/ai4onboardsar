"""Download GRD products (label source) and matching RAW L0 products (model input) for all scenes.

Usage: python scripts/download.py [--config configs/scenes.yaml] [--scenes s4_20250422 ...]
"""
import argparse
import json
import sys
from pathlib import Path

import yaml
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai4onboardsar.cdse import CDSE  # noqa: E402


def needed_raw(raw: dict, grd_names: list, margin_s: float = 1.0) -> dict:
    """Greedy cover of the GRD sensing interval (+margin) with as few RAW slices as possible."""
    fmt = "%Y%m%dT%H%M%S"
    t0 = min(datetime.strptime(n.split("_")[4], fmt) for n in grd_names) - timedelta(seconds=margin_s)
    t1 = max(datetime.strptime(n.split("_")[5], fmt) for n in grd_names) + timedelta(seconds=margin_s)
    span = {n: (datetime.strptime(n.split("_")[5], fmt), datetime.strptime(n.split("_")[6], fmt)) for n in raw}
    # the margin is best effort (the datatake may start/end there)
    t0 = max(t0, min(a for a, _ in span.values()))
    t1 = min(t1, max(b for _, b in span.values()))
    chosen, covered = [], t0
    while covered < t1:
        cands = [n for n, (a, b) in span.items() if a <= covered and b > covered]
        if not cands:
            raise RuntimeError(f"no RAW slice covers {covered}")
        best = max(cands, key=lambda n: span[n][1])
        chosen.append(best)
        covered = span[best][1]
    return {n: raw[n] for n in chosen}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/scenes.yaml")
    ap.add_argument("--scenes", nargs="*")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    root = Path(cfg["data_root"])
    client = CDSE()

    for scene_id, scene in cfg["scenes"].items():
        if args.scenes and scene_id not in args.scenes:
            continue
        print(f"== {scene_id}")
        raw = {}
        for grd_name in scene["grd"]:
            grd = client.by_name(grd_name)
            print(f"  GRD {grd_name} ({grd['ContentLength'] / 1e6:.0f} MB)")
            client.download(grd, root / "grd")
            for r in client.matching_raw(grd_name):
                raw[r["Name"]] = r
        raw = needed_raw(raw, scene["grd"])
        for name, r in raw.items():
            print(f"  RAW {name} ({r['ContentLength'] / 1e6:.0f} MB)")
            client.download(r, root / "raw")
        manifest = {"grd": scene["grd"], "raw": sorted(raw)}
        (root / "scenes").mkdir(parents=True, exist_ok=True)
        json.dump(manifest, open(root / "scenes" / f"{scene_id}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
