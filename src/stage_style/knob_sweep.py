"""M6 -- knob-sweep validation.

For each style knob, sweep it 0->1 (others held at the reference) and measure the
emergent behaviour it is meant to control, averaged over the tail of a short
rollout. A working knob moves its metric monotonically.

  line height -> team0 outfield mean forward-x (higher = higher line)
  press       -> mean team0 player-to-ball distance (lower = more press)
  width       -> team0 outfield y-spread (higher = wider)
  tempo       -> mean ball speed (higher = faster)

    python -m stage_style.knob_sweep
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from stage8_sim.simulate_gsr import Sim, N_SLOTS, BALL

ROOT = Path(__file__).resolve().parents[2]


def _seed(store):
    d = np.load(Path(store) / "windows_val.npz")
    i = int(d["presence"][:, -1, :22].sum(-1).argmax())   # highest-visibility window
    return {k: d[k][i] for k in ("agents", "presence", "style0", "style1")}


def rollout_metrics(model_path, knob_idx, val, steps=70, tail=25, seed=None):
    sim = Sim(model_path, seed=seed)
    base = sim.knobs.copy() if seed is not None else sim.ref.cpu().numpy()[0].copy()
    sim.knobs = base
    sim.knobs[knob_idx] = val
    fwdx, p2b, yspread, bspeed = [], [], [], []
    prev_ball = sim.pos[BALL].copy()
    for s in range(steps):
        sim.step()
        of = sim.pos[1:N_SLOTS]                       # team0 outfield
        fwdx.append(float(of[:, 0].mean()))           # team0 attacks +x
        p2b.append(float(np.linalg.norm(sim.pos[:N_SLOTS] - sim.pos[BALL], axis=1).mean()))
        yspread.append(float(np.percentile(of[:, 1], 90) - np.percentile(of[:, 1], 10)))
        bspeed.append(float(np.linalg.norm(sim.pos[BALL] - prev_ball) * sim.fps))
        prev_ball = sim.pos[BALL].copy()
    sl = slice(-tail, None)
    return {"line": np.mean(fwdx[sl]), "press": np.mean(p2b[sl]),
            "width": np.mean(yspread[sl]), "tempo": np.mean(bspeed[sl])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default=str(ROOT / "output" / "training" / "metrica" / "style_model.pt"))
    ap.add_argument("--store", default=str(ROOT / "output" / "storage" / "metrica"))
    ap.add_argument("--steps", type=int, default=70)
    args = ap.parse_args()
    vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    knob_metric = {0: ("line", "line-height -> mean fwd-x (m), expect UP"),
                   1: ("press", "press -> player-to-ball (m), expect DOWN"),
                   2: ("width", "width -> y-spread (m), expect UP"),
                   3: ("tempo", "tempo -> ball speed (m/s), expect UP")}
    seed = _seed(args.store)
    print("knob-sweep (seeded from real window; others held at the clip's style):\n")
    summary = {}
    for k, (metric, desc) in knob_metric.items():
        row = []
        for v in vals:
            m = rollout_metrics(args.model, k, v, args.steps, seed=seed)
            row.append(m[metric])
        # monotonicity (Spearman sign via rank correlation with vals)
        rank = np.argsort(np.argsort(row))
        corr = np.corrcoef(np.arange(len(vals)), rank)[0, 1]
        summary[metric] = corr
        arrow = "UP" if corr > 0 else "DOWN"
        print(f"{desc}")
        print("   knob:  " + "  ".join(f"{v:.2f}" for v in vals))
        print("   metric:" + "  ".join(f"{x:.2f}" for x in row))
        print(f"   trend: {arrow} (rank-corr {corr:+.2f})\n")
    print("monotonic & correct-direction knobs:")
    want = {"line": +1, "press": -1, "width": +1, "tempo": +1}
    for m, c in summary.items():
        ok = "YES" if np.sign(c) == want[m] and abs(c) >= 0.6 else "weak"
        print(f"   {m:6s}: {ok}")


if __name__ == "__main__":
    main()
