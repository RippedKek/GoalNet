"""Ablation runs: remove one component at a time, measure the football.

Proof that the learned models are playing the game: if KickNet or StyleNet
is switched off and the match statistics collapse, the statistics were
coming from the models.

  cd src && python -m stage8_sim.ablation --steps 3000

Configs
  full            hybrid as shipped
  random-kick     KickNet replaced by random action + uniform receiver
  frozen-players  StyleNet player deltas zeroed (structure layer only)
  no-structure    anchors/press layer off (pure StyleNet + ball engine)
  high-press      full, team0 press knob 0.95 (FiLM + mechanical dial)
  high-tempo      full, team0 tempo knob 0.95 (FiLM + pass directness)

Writes per-config telemetry npz, a summary json, comparison figures, and
the full-config evidence figures to output/telemetry/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stage8_sim.telemetry import COLS, make_match_figures, _load, _roll, \
    BLUE, AQUA, YELLOW, RED, GRAY

ROOT = Path(__file__).resolve().parents[2]

CONFIGS = {
    "full":           {},
    "random-kick":    {"random_kick": True},
    "frozen-players": {"disable_stylenet": True},
    "no-structure":   {"disable_structure": True},
    "high-press":     {"knobs": {1: 0.95}},
    "high-tempo":     {"knobs": {3: 0.95}},
}


def run_config(name, cfg, args):
    from stage8_sim.simulate_gsr import Sim
    sim = Sim(args.model, seed=None, kick_path=args.kick)
    sim.set_formations(args.formation0, args.formation1)
    sim.apply_kickoff(0)
    sim.fast = True
    sim.disable_stylenet = cfg.get("disable_stylenet", False)
    sim.disable_structure = cfg.get("disable_structure", False)
    sim.engine.random_kick = cfg.get("random_kick", False)
    for k, v in cfg.get("knobs", {}).items():
        sim.knobs[k] = v
    for s in range(args.steps):
        sim.step()
        if sim.goal_pending:                       # kick-off after every goal
            sim.goal_pending = False
            sim.apply_kickoff(1 - (sim.engine.last_goal_team or 0))
        if s % 50 == 25 and not sim.disable_stylenet:
            sim.knob_sensitivity()                 # FiLM authority probe
    return sim


def summarize(name, sim, steps):
    mins = steps / sim.fps / 60.0
    st = sim.stats
    A = sim.telemetry.array()
    c = {n: i for i, n in enumerate(COLS)}
    poss = A[:, c["poss"]]
    v = poss[poss >= 0]
    swaps = float((v[1:] != v[:-1]).sum()) if len(v) > 1 else 0.0
    att = st.t[0]["pass_att"] + st.t[1]["pass_att"]
    cmp_ = st.t[0]["pass_cmp"] + st.t[1]["pass_cmp"]
    return {
        "config": name, "mins": mins,
        "passes_min": att / mins,
        "cmp_pct": 100.0 * cmp_ / max(att, 1),
        "shots10": 10.0 * (st.t[0]["shots"] + st.t[1]["shots"]) / mins,
        "goals": st.t[0]["goals"] + st.t[1]["goals"],
        "int10": 10.0 * (st.t[0]["intercepts"] + st.t[1]["intercepts"]) / mins,
        "tackles10": 10.0 * (st.t[0]["tackles"] + st.t[1]["tackles"]) / mins,
        "swaps_min": swaps / mins,
        "player_mps": float(np.nanmean(A[:, c["player_mps"]])),
        "x_spread": float(np.nanmean(A[:, c["x_spread"]])),
        "poss0": st.possession()[0],
        "stats": st.as_dict(),
    }


def fig_compare(summaries, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = [("passes_min", "passes / min"), ("cmp_pct", "pass completion %"),
               ("shots10", "shots / 10 min"), ("int10", "interceptions / 10 min"),
               ("swaps_min", "possession swaps / min"),
               ("player_mps", "mean player speed (m/s)")]
    names = list(summaries)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7), dpi=140)
    fig.suptitle("Ablation: what breaks when each component is removed "
                 "(green = full hybrid)")
    for ax, (k, lab) in zip(axes.flat, metrics):
        vals = [summaries[n][k] for n in names]
        cols = [AQUA if n == "full" else
                (RED if n in ("random-kick", "frozen-players", "no-structure")
                 else BLUE) for n in names]
        ax.bar(range(len(vals)), vals, color=cols)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=7.5)
        ax.set_title(lab, fontsize=9)
    fig.tight_layout()
    p = Path(outdir) / "abl_compare.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_timelines(npz_paths, outdir, fps, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(npz_paths), 1, figsize=(11, 1.9 * len(npz_paths)),
                             dpi=140, sharex=True, sharey=True)
    w = int(30 * fps)
    for ax, (name, path) in zip(np.atleast_1d(axes), npz_paths.items()):
        A, c = _load(path)
        t = A[:, c["tick"]] / fps / 60.0
        poss = A[:, c["poss"]]
        p0 = (poss == 0).astype(np.float32)
        p1 = (poss == 1).astype(np.float32)
        share0 = _roll(p0, w) / np.maximum(_roll(p0 + p1, w), 1e-6)
        ax.plot(t, share0, color=BLUE, lw=1.2)
        ax.axhline(0.5, color=GRAY, lw=0.7, ls=":")
        ax.fill_between(t, 0.5, share0, where=share0 >= 0.5, color=BLUE, alpha=0.18)
        ax.fill_between(t, share0, 0.5, where=share0 < 0.5, color=YELLOW, alpha=0.28)
        ax.set_ylim(0, 1)
        ax.set_ylabel(name, fontsize=8)
    np.atleast_1d(axes)[0].set_title(
        f"{names[0]} possession share (30 s rolling) per ablation config")
    np.atleast_1d(axes)[-1].set_xlabel("match minutes")
    fig.tight_layout()
    p = Path(outdir) / "abl_timelines.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default=str(ROOT / "output" / "training" / "metrica" / "style_model.pt"))
    ap.add_argument("--kick", default=str(ROOT / "output" / "kick" / "kick_model.pt"))
    ap.add_argument("--steps", type=int, default=3000,
                    help="ticks per config (3000 = 10 sim-minutes)")
    ap.add_argument("--formation0", default="4-4-2")
    ap.add_argument("--formation1", default="4-3-3")
    ap.add_argument("--configs", default=",".join(CONFIGS),
                    help="comma list from: " + ", ".join(CONFIGS))
    ap.add_argument("--out", default=str(ROOT / "output" / "telemetry"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summaries, npz_paths, team_names = {}, {}, ("team0", "team1")
    for name in args.configs.split(","):
        name = name.strip()
        print(f"=== {name} ({args.steps} steps) ===")
        sim = run_config(name, CONFIGS[name], args)
        team_names = tuple(sim.stats.names)
        npz_paths[name] = sim.telemetry.save(out / f"abl_{name}.npz")
        summaries[name] = summarize(name, sim, args.steps)
        s = summaries[name]
        print(f"  pass/min {s['passes_min']:.1f}  cmp {s['cmp_pct']:.0f}%  "
              f"shots/10 {s['shots10']:.1f}  goals {s['goals']}  "
              f"int/10 {s['int10']:.1f}  swaps/min {s['swaps_min']:.1f}  "
              f"speed {s['player_mps']:.2f} m/s  xspread {s['x_spread']:.1f} m")
        print(sim.stats.table_str())

    (out / "ablation_summary.json").write_text(json.dumps(summaries, indent=2))
    figs = [fig_compare(summaries, out),
            fig_timelines(npz_paths, out, fps=5, names=team_names)]
    if "full" in npz_paths:
        figs += make_match_figures(npz_paths["full"], out, fps=5,
                                   names=team_names, prefix="abl_full")
    print("\nwrote:", out / "ablation_summary.json")
    for f in figs:
        print("fig:", f)

    hdr = (f"{'config':16}{'pass/min':>9}{'cmp%':>6}{'shots/10':>9}{'goals':>6}"
           f"{'int/10':>7}{'swaps/min':>10}{'m/s':>6}{'xsprd':>7}{'poss0':>7}")
    print("\n" + hdr)
    for n, s in summaries.items():
        print(f"{n:16}{s['passes_min']:9.1f}{s['cmp_pct']:6.0f}{s['shots10']:9.1f}"
              f"{s['goals']:6d}{s['int10']:7.1f}{s['swaps_min']:10.1f}"
              f"{s['player_mps']:6.2f}{s['x_spread']:7.1f}{s['poss0'] * 100:6.0f}%")


if __name__ == "__main__":
    main()
