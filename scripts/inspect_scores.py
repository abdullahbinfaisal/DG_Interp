r"""
Inspect a concept score file: integrity checks and population sizes.

  & $PY scripts\inspect_scores.py processed\FINAL_ERM_ResNet_3300_T3.json

Reports which scores are present, the number of source domains the scores were
estimated over, sentinel/degenerate value counts, and the H x R contingency
tables under each D regime -- the populations that determine whether a given
threshold choice leaves viable bucket sizes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CLASS_NAMES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]


def load(path: Path) -> dict:
    # Read as bytes then decode: text-mode reads of these 50MB+ files have
    # produced spurious JSONDecodeErrors on this machine.
    with open(path, "rb") as fh:
        return json.loads(fh.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--support-floor", type=int, default=30,
                    help="H_counts minimum for the statistics population")
    ap.add_argument("--tau-d", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, nargs="*", default=[0.7, 0.8, 0.9],
                    help="tau_H = tau_R values to tabulate")
    args = ap.parse_args()

    path = Path(args.path)
    data = load(path)

    print(f"file        : {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"classes     : {len(data)}")
    sample = data["0"]["0"]
    print(f"scores      : {sorted(sample.keys())}")
    n_dom = len(sample.get("R_acts", []))
    print(f"domains in R_acts : {n_dom}"
          f"{'  <-- SOURCE ONLY, correct' if n_dom == 3 else '  <-- WARNING: expected 3'}")

    # ---- collect ----------------------------------------------------------
    rows = []
    for k in range(7):
        for c, v in data[str(k)].items():
            rows.append((k, int(c), v))

    n = len(rows)
    r_sentinel = sum(1 for _, _, v in rows if v.get("R") == -1.0)
    r_zero = sum(1 for _, _, v in rows if v.get("R") == 0.0)
    mixed = 0
    for _, _, v in rows:
        ra = v.get("R_acts") or []
        if any(x > 0 for x in ra) and any(x < 0 for x in ra):
            mixed += 1

    print()
    print(f"pairs total          : {n}")
    print(f"R == -1 (legacy sentinel): {r_sentinel}"
          f"{'  <-- OK, magnitude R' if r_sentinel == 0 else '  <-- STALE signed-R FILE'}")
    print(f"R == 0 (inert)       : {r_zero}")
    print(f"sign-flipping R_acts : {mixed}  ({mixed / n * 100:.2f}% of all pairs)")

    # ---- statistics population -------------------------------------------
    floor = args.support_floor
    sup = [(k, c, v) for k, c, v in rows if v.get("H_counts", 0) >= floor]
    print()
    print(f"--- statistics population (H_counts >= {floor}) : {len(sup)} pairs ---")

    tau_d = args.tau_d
    regimes = [
        (f"neutral   |D| <= {tau_d:g}", lambda d: abs(d) <= tau_d),
        (f"harmful   D  < -{tau_d:g}", lambda d: d < -tau_d),
        (f"supportive D >  {tau_d:g}", lambda d: d > tau_d),
    ]

    for tau in args.tau:
        print()
        print(f"=== tau_H = tau_R = {tau} , tau_D = {tau_d:g} ===")
        for label, test in regimes:
            cells = {(hi, ri): 0 for hi in (0, 1) for ri in (0, 1)}
            for _, _, v in sup:
                if not test(v["D"]):
                    continue
                hi = 1 if v["H"] >= tau else 0
                ri = 1 if v["R"] >= tau else 0
                cells[(hi, ri)] += 1
            tot = sum(cells.values())
            print(f"  {label}   n={tot}")
            if tot == 0:
                continue
            print(f"{'':>14}{'R<tau':>10}{'R>=tau':>10}")
            for hi, hlabel in ((0, "H<tau"), (1, "H>=tau")):
                lo, hh = cells[(hi, 0)], cells[(hi, 1)]
                print(f"{hlabel:>14}{lo:>10}{hh:>10}"
                      f"   ({lo / tot * 100:5.1f}% / {hh / tot * 100:5.1f}%)")

        # The claim that matters for section 5.3.
        hi_h_pos = [v for _, _, v in sup if v["D"] > tau_d and v["H"] >= tau]
        hi_h_neg = [v for _, _, v in sup if v["D"] < -tau_d and v["H"] >= tau]
        for lbl, grp in (("supportive", hi_h_pos), ("harmful", hi_h_neg)):
            if grp:
                frac = sum(1 for v in grp if v["R"] >= tau) / len(grp) * 100
                print(f"  of high-H {lbl:>10} pairs, {frac:5.1f}% are also high-R "
                      f"({sum(1 for v in grp if v['R'] >= tau)}/{len(grp)})")

    # ---- per class --------------------------------------------------------
    print()
    print(f"--- per-class non-neutral counts (support >= {floor}, |D| > {tau_d:g}) ---")
    print(f"{'class':>10}{'support':>10}{'D>0':>8}{'D<0':>8}")
    for k in range(7):
        s = [v for kk, _, v in sup if kk == k]
        pos = sum(1 for v in s if v["D"] > tau_d)
        neg = sum(1 for v in s if v["D"] < -tau_d)
        print(f"{CLASS_NAMES[k]:>10}{len(s):>10}{pos:>8}{neg:>8}")


if __name__ == "__main__":
    main()
